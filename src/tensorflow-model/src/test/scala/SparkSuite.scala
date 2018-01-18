// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.ml.spark

import org.apache.spark.sql.Row
import org.apache.spark.sql.catalyst.encoders.RowEncoder
import org.apache.spark.sql.functions.col
import org.apache.spark.sql.types.{StringType, StructField, StructType}
import org.apache.spark.sql._
import org.apache.spark.ml.linalg.DenseVector
import com.microsoft.ml.spark.Readers.implicits._
import org.apache.spark

import scala.collection.mutable.ArrayBuffer


class SparkSuite extends TestBase{

  test("Test 1: Loading images to Spark DF and performing a basic operation") {
    //Start of setup - code repeated in tests
    val enc = RowEncoder(new StructType().add(StructField("new col", StringType)))
    val inputCol = "images"
    val outputCol = "out"

    val filesRoot = s"${sys.env("DATASETS_HOME")}/"
    val imagePath = s"$filesRoot/Images/CIFAR"
    val images = session.readImages(imagePath, true)
    val unroll = new UnrollImage().setInputCol("image").setOutputCol(inputCol)

    val processed_images = unroll.transform(images).select(inputCol)
    processed_images.show()
    //End of setup

    val processed_images_tf = processed_images.mapPartitions { it =>
      it.map { r =>
        val row = Row.fromSeq(Array(r.length.toString).toSeq)
        row
      }
    }(enc)

    processed_images_tf.show()
  }

  test("Test 2: Loading image to DF and changing the numerical values"){
    //Start of setup - code repeated in tests
    val enc = RowEncoder(new StructType().add(StructField("new col", StringType)))
    val inputCol = "cntk_images"
    val outputCol = "out"

    val filesRoot = s"${sys.env("DATASETS_HOME")}/"
    val imagePath = s"$filesRoot/Images/CIFAR"
    val images = session.readImages(imagePath, true)
    val unroll = new UnrollImage().setInputCol("image").setOutputCol(inputCol)

    val processed_images = unroll.transform(images).select(inputCol)
    processed_images.show()
    //End of setup

    val processed_images_tf = processed_images.mapPartitions { it =>
      it.map { r =>
        val rawData = r.toSeq.toArray
        val rawDataDouble: Seq[Double] = rawData(0).asInstanceOf[DenseVector].values.toSeq
        //for above - TODO: Am I sure this is always the case? --> type and containing one element
        val transformed = rawDataDouble.map(_ + 10.0).map(t => t.toString)
//        val arrayTransformed = Array(transformed)
        Row.fromSeq(transformed) //contains multiple elements, need to be changed into Seq of one denseVector
      }

    }(enc)

    processed_images_tf.show()

  }
//    val df = processed_images.mapPartitions { it =>
//      println("Only once")
//      it.map {r =>
//        r.size
//      }
//    }(enc)


//  test("foo"){
//    val enc = RowEncoder(new StructType().add(StructField("new col", StringType)))
//    val df = session
//      .createDataFrame(Seq((1,2),(2,3),(4,3)))
//      .withColumn("foo",col("_2")+2)
//      .mapPartitions { it =>
//        println("i only run once")
//        val model = ??? //loadme!
//        it.map {r =>
//          val rawData = r.getAs[Array[Double]](???)
//          val tfData = toTFData(rawData)
//          val tfResults = model.eval(tfData)
//          val sparkResults = fromTFData(tfResults)
//        }
//      }(enc)
//    df.show()
//    df.printSchema()
//  }
//
//  test("bar"){
//    import session.implicits._
//
//    sc.parallelize((1 to 10).map(Tuple1(_))).toDF("foo").rdd.mapPartitions()
//  }

}
