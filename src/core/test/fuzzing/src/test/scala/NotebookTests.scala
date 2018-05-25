// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.ml.spark

import com.microsoft.ml.spark.DatabricksUtilities._

import scala.concurrent.Await
import scala.concurrent.duration.Duration
import scala.language.existentials

/** Tests to validate fuzzing of modules. */
class NotebookTests extends TestBase {

  test("Databricks Notebooks") {
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

}
