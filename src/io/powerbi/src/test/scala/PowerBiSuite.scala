// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.ml.spark

import org.apache.spark.SparkException
import org.apache.spark.sql.catalyst.encoders.RowEncoder
import org.apache.spark.sql.functions.{current_timestamp, lit}

import scala.collection.JavaConverters._

class PowerBiSuite extends TestBase with FileReaderUtils {

  lazy val url = sys.env("MML_POWERBI_URL")
  lazy val df = session
    .createDataFrame(Seq(
      (Some(0), "a"),
      (Some(1), "b"),
      (Some(2), "c"),
      (Some(3), ""),
      (None, "bad_row")))
    .toDF("bar", "foo")
    .withColumn("baz", current_timestamp())
  lazy val bigdf = (1 to 5).foldRight(df) { case (_, ldf) => ldf.union(df) }.repartition(2)
  lazy val delayDF = {
    val rows = Array.fill(100){df.collect()}.flatten.toList.asJava
    val df2 = session
      .createDataFrame(rows, df.schema)
      .coalesce(1).cache()
    df2.count()
    df2.map({x => Thread.sleep(10); x})(RowEncoder(df2.schema))
  }

  lazy val bigFastDF = {
    val rows = Array.fill(10000){df.collect()}.flatten.toList.asJava
    val df3 = session
      .createDataFrame(rows, df.schema)
      .cache()
    df3.count()
    df3
  }

  test("write to powerBi", TestBase.BuildServer) {
    PowerBIWriter.write(df, url)
  }

  test("write to powerBi with delays"){
    PowerBIWriter.write(delayDF, url)
  }

  test("using dynamic minibatching"){
    PowerBIWriter.write(delayDF, url, Map("minibatcher"->"dynamic"))
  }

  test("using timed minibatching"){
    PowerBIWriter.write(delayDF, url, Map("minibatcher"->"timed"))
  }

  test("using consolidated timed minibatching"){
    PowerBIWriter.write(delayDF, url, Map(
      "minibatcher"->"timed",
      "consolidate"->"true"))
  }

  test("using buffered batching"){
    PowerBIWriter.write(delayDF, url, Map("buffered"->"true"))
  }

  ignore("throw useful error message when given an improper dataset") {
    //TODO figure out why this does not throw errors on the build machine
    assertThrows[SparkException] {
      PowerBIWriter.write(df.withColumn("bad", lit("foo")), url)
    }
  }

  test("stream to powerBi", TestBase.BuildServer) {
    bigdf.write.parquet(tmpDir + "powerBI.parquet")
    val sdf = session.readStream.schema(df.schema).parquet(tmpDir + "powerBI.parquet")
    val q1 = PowerBIWriter.stream(sdf, url).start()
    q1.processAllAvailable()
  }

}
