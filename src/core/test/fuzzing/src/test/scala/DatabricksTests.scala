// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.ml.spark

import org.apache.commons.io.IOUtils
import org.spark_project.guava.io.BaseEncoding
import org.apache.http.client.methods._
import org.apache.http.entity.StringEntity
import org.apache.http.impl.client.{CloseableHttpClient, HttpClientBuilder}
import spray.json._
import spray.json.DefaultJsonProtocol._

import scala.language.existentials

/** Tests to validate fuzzing of modules. */
class DatabricksTests extends TestBase {

  val client: CloseableHttpClient = HttpClientBuilder.create().build()
  val region = "eastus2"
  val token = sys.env("MML_ADB_TOKEN")
  val authValue: String = "Basic " + BaseEncoding.base64().encode(("token:" + token).getBytes("UTF-8"))

  val clusterName = "Test Cluster 2"
  lazy val clusterId: String = getClusterIdByName(clusterName)
  val version = "com.microsoft.ml.spark:mmlspark_2.11:0.12.dev9+5.ge162a0c"
  val baseURL = s"https://$region.azuredatabricks.net/api/2.0/"
  val libraryString: String = Map("maven" -> Map(
    "coordinates" -> version,
    "repo" -> "https://mmlspark.azureedge.net/maven"
  )).toJson.compactPrint

  val timeoutInMillis: Int = 10 * 60 * 1000

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

  def submitJob(notebookPath: String, timeout: Int = 10 * 60): Int = {
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

  def monitorJob(runId: Integer,
                 timeout: Int = timeoutInMillis,
                 interval: Int = 1000, quiet: Boolean = true): Unit = {
    val start = System.currentTimeMillis()
    val runObj = databricksGet(s"jobs/runs/get?run_id=$runId")
    val runUrl = runObj.asJsObject.fields("run_page_url").asInstanceOf[JsString].value
    while (System.currentTimeMillis() - start < timeout) {
      val runObj = databricksGet(s"jobs/runs/get?run_id=$runId")
      val stateObj = runObj.asJsObject.fields("state").asJsObject
      val lifeCycleState = stateObj.fields("life_cycle_state").asInstanceOf[JsString].value
      if (lifeCycleState == "TERMINATED") {
        val resultState = stateObj.fields("result_state").asInstanceOf[JsString].value
        resultState match {
          case "SUCCESS" =>
            println(s"Run $runId suceeded")
            return
          case _ => throw new RuntimeException(s"Job failed, see $runUrl for stack trace")
        }
      } else {
        if (!quiet) println(s"STATE: $lifeCycleState, object: $runObj")
      }
      Thread.sleep(interval.toLong)
    }
    throw new RuntimeException(s"Job did not finish before timeout, see $runUrl for stack trace")
  }

  test("Submit failing job") {
    val jobNum = submitJob("/SampleJobs/Job1")
    assertThrows[RuntimeException](monitorJob(jobNum))
  }

  test("Submit successful job") {
    val jobNum = submitJob("/SampleJobs/Job2")
    monitorJob(jobNum, quiet = false)
  }

  test("get library status") {
    val libraryObj = databricksGet(s"libraries/cluster-status?cluster_id=$clusterId")
    println(libraryObj)
  }

  test("get jobs") {
    val jobs = databricksGet("jobs/runs/list")
    println(jobs)
  }

}
