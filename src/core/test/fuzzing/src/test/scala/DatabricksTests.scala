// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.ml.spark

import java.io.FileInputStream
import java.util.concurrent.TimeoutException

import com.microsoft.ml.spark.FileUtilities.File
import org.apache.commons.io.IOUtils
import org.apache.http.client.methods._
import org.apache.http.entity.StringEntity
import org.apache.http.impl.client.{CloseableHttpClient, HttpClientBuilder}
import org.spark_project.guava.io.BaseEncoding
import spray.json.DefaultJsonProtocol._
import spray.json._

import scala.concurrent.duration.Duration
import scala.concurrent.{Await, ExecutionContext, Future, blocking}
import scala.language.existentials
import scala.util.{Failure, Success}


/** Tests to validate fuzzing of modules. */
class DatabricksTests extends TestBase {

  lazy val client: CloseableHttpClient = HttpClientBuilder.create().build()

  // ADB Info
  val region = "eastus2"
  val token = sys.env("MML_ADB_TOKEN")
  val authValue: String = "Basic " + BaseEncoding.base64().encode(("token:" + token).getBytes("UTF-8"))
  val clusterName = "Test Cluster 2"
  lazy val clusterId: String = getClusterIdByName(clusterName)
  val folder = "/MMLSparkBuild/Build1"

  //MMLSpark info
  val version = "com.microsoft.ml.spark:mmlspark_2.11:0.12.dev9+5.ge162a0c"
  val baseURL = s"https://$region.azuredatabricks.net/api/2.0/"
  val libraryString: String = Map("maven" -> Map(
    "coordinates" -> version,
    "repo" -> "https://mmlspark.azureedge.net/maven"
  )).toJson.compactPrint

  //Execution Params
  val timeoutInMillis: Int = 10 * 60 * 1000

  val notebookFiles: Array[File] = Option(
    new File("./BuildArtifacts/notebooks/hdinsight").getCanonicalFile.listFiles()
  ).get

  def databricksGet(path: String): JsValue = {
    val request = new HttpGet(baseURL + path)
    request.addHeader("Authorization", authValue)
    val response = client.execute(request)
    if (response.getStatusLine.getStatusCode != 200) {
      throw new RuntimeException(s"Failed: response: $response")
    }
    IOUtils.toString(response.getEntity.getContent).parseJson
  }

  //TODO convert all this to typed code
  def databricksPost(path: String, body: String): JsValue = {
    val request = new HttpPost(baseURL + path)
    request.addHeader("Authorization", authValue)
    request.setEntity(new StringEntity(body))
    val response = client.execute(request)

    if (response.getStatusLine.getStatusCode != 200) {
      val entity = IOUtils.toString(response.getEntity.getContent, "UTF-8")
      throw new RuntimeException(s"Failed:\n entity:$entity \n response: $response")
    }
    IOUtils.toString(response.getEntity.getContent).parseJson
  }

  def getClusterIdByName(name: String): String = {
    val jsonObj = databricksGet("clusters/list")
    val cluster = jsonObj.asJsObject
      .fields("clusters").asInstanceOf[JsArray].elements
      .filter(_.asJsObject
        .fields("cluster_name").asInstanceOf[JsString].value == name).head
    cluster.asJsObject.fields("cluster_id").asInstanceOf[JsString].value
  }

  def workspaceMkDir(dir: String): Unit = {
    val body = s"""{"path": "$dir"}"""
    databricksPost("workspace/mkdirs", body)
    ()
  }

  def uploadNotebook(file: File, dest: String): Unit = {
    val content = BaseEncoding.base64().encode(
      IOUtils.toByteArray(new FileInputStream(file)))
    val body =
      s"""
         |{
         |  "content": "$content",
         |  "path": "$dest",
         |  "overwrite": true,
         |  "format": "JUPYTER"
         |}
       """.stripMargin
    databricksPost("workspace/import", body)
    ()
  }

  def workspaceRmDir(dir: String): Unit = {
    val body = s"""{"path": "$dir", "recursive":true}"""
    databricksPost("workspace/delete", body)
    ()
  }

  def submitRun(notebookPath: String, timeout: Int = 10 * 60): Int = {
    val body =
      s"""
         |{
         |  "run_name": "test1",
         |  "existing_cluster_id": "$clusterId",
         |  "timeout_seconds": ${timeoutInMillis / 1000},
         |  "notebook_task": {
         |    "notebook_path": "$notebookPath",
         |    "base_parameters": []
         |  },
         |  "libraries": [$libraryString]
         |}
      """.stripMargin

    databricksPost("jobs/runs/submit", body)
      .asJsObject().fields("run_id").asInstanceOf[JsNumber].value.toInt
  }

  private def getRunStatuses(runId: Int): (String, Option[String]) = {
    val runObj = databricksGet(s"jobs/runs/get?run_id=$runId")
    val stateObj = runObj.asJsObject.fields("state").asJsObject
    val lifeCycleState = stateObj.fields("life_cycle_state").asInstanceOf[JsString].value
    if (lifeCycleState == "TERMINATED") {
      val resultState = stateObj.fields("result_state").asInstanceOf[JsString].value
      (lifeCycleState, Some(resultState))
    } else {
      (lifeCycleState, None)
    }
  }

  def getRunUrlAndNBName(runId: Int): (String, String) = {
    val runObj = databricksGet(s"jobs/runs/get?run_id=$runId").asJsObject()
    val url = runObj.fields("run_page_url")
      .asInstanceOf[JsString].value
    val nbName = runObj.fields("task").asJsObject()
      .fields("notebook_task").asJsObject
      .fields("notebook_path").convertTo[String]
    (url, nbName)
  }

  def monitorJob(runId: Integer,
                 interval: Int = 2000,
                 timeout: Int = 10000,
                 logLevel: Int = 1): Future[Unit] = {
    Future {
      var finalState: Option[String] = None
      var lifeCycleState: String = "Not Started"
      val startTime = System.currentTimeMillis()
      while (finalState.isEmpty & (System.currentTimeMillis() - startTime) < timeout) {
        val (lcs, fs) = getRunStatuses(runId)
        finalState = fs
        lifeCycleState = lcs
        if (logLevel >= 2) println(s"Job $runId state: $lifeCycleState")
        blocking {
          Thread.sleep(interval.toLong)
        }
      }
      val (url, nbName) = getRunUrlAndNBName(runId)

      val error = finalState match {
        case Some("SUCCESS") =>
          if (logLevel >= 1) println(s"Notebook $nbName Suceeded")
          None
        case Some(state) =>
          Some(new RuntimeException(s"Notebook $nbName failed with state $state. " +
            s"For more information check the run page: \n$url\n"))
        case None =>
          Some(new TimeoutException(s"Notebook $nbName timed out after $timeout ms," +
            s" job in state $lifeCycleState, " +
            s" For more information check the run page: \n$url\n "))
      }

      error.foreach { error =>
        if (logLevel >= 1) print(error.getMessage)
        throw error
      }

    }(ExecutionContext.global)
  }

  def uploadAndSubmitNotebook(notebookFile: File): Int = {
    uploadNotebook(notebookFile, folder + "/" + notebookFile.getName)
    submitRun(folder + "/" + notebookFile.getName)
  }

  def cancelRun(runId: Int): Unit = {
    databricksPost("jobs/runs/cancel", s"""{"run_id": $runId}""")
    ()
  }

  def cancelAllJobs(clusterId: String): Unit = {
    //TODO this only gets the first 1k running jobs, full solution would page results
    databricksGet("jobs/runs/list?active_only=true&limit=1000")
      .asJsObject.fields.get("runs").foreach { runs =>
      runs.asInstanceOf[JsArray].elements.foreach { e =>
        val cid = e.asJsObject.fields("cluster_instance").asJsObject.fields("cluster_id").convertTo[String]
        if (cid == clusterId) {
          val rid = e.asJsObject.fields("run_id").convertTo[Int]
          println(s"Cancelling run $rid")
          cancelRun(rid)
        }
      }
    }
  }

  test("Submit failing job") {
    val jobNum = submitRun("/SampleJobs/Job1")
    assertThrows[RuntimeException](monitorJob(jobNum))
  }

  test("Submit successful job") {
    val jobNum = submitRun("/SampleJobs/Job2")
    monitorJob(jobNum)
  }

  test("get library status") {
    val libraryObj = databricksGet(s"libraries/cluster-status?cluster_id=$clusterId")
    println(libraryObj)
  }

  test("get jobs") {
    val jobs = databricksGet("jobs/runs/list")
    println(jobs)
  }

  test("Upload and run jobs") {
    workspaceRmDir(folder)
    workspaceMkDir(folder)
    val jobIds = notebookFiles.map(uploadAndSubmitNotebook)
    println(s"Submitted ${jobIds.length} for execution")
    try {
      val monitors = jobIds.map(monitorJob(_, timeout = timeoutInMillis))
      println(s"Monitoring Jobs...")
      val failures = monitors
        .map(Await.ready(_, Duration.Inf).value.get)
        .filter(_.isFailure)
      assert(failures.isEmpty)
    } catch {
      case t: Throwable =>
        jobIds.foreach { jid =>
          println(s"Cancelling job $jid")
          cancelRun(jid)
        }
        throw t
    }
  }

  test("clean up") {
    workspaceRmDir(folder)
  }

  test("cancel all jobs") {
    cancelAllJobs(clusterId)
  }

}
