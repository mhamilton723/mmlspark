// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.ml.spark

import org.apache.commons.io.IOUtils
import org.tensorflow.{DataType, Graph, Output, Session, Shape, Tensor, TensorFlow => tf}
import org.tensorflow.types.UInt8


class TensorflowTrainerSuite extends TestBase {


  test("Modify a graph") {
    val g = new Graph
    val b = new TensorflowGraphBuilder(g)
    val s = new Session(g)

    val W = b.variable("W", Shape.scalar(), classOf[java.lang.Double])
    val W_init = b.assign(b.constant("Wval", 1.0), W)
    val W_init2 = b.assign(b.constant("Wval2", 2.0), W)

    val result = s.runner().fetch(W_init).run().get(0)
    val resultW = s.runner().fetch(W).run().get(0)
    assert(result.doubleValue() === 1)
    assert(resultW.doubleValue() === 1)

    val result2 = s.runner().fetch(W_init2).run().get(0)
    val resultW2 = s.runner().fetch(W).run().get(0)
    assert(result2.doubleValue() === 2)
    assert(resultW2.doubleValue() === 2)
  }

  test("placeholders") {
    val g = new Graph
    val b = new TensorflowGraphBuilder(g)
    val s = new Session(g)

    val x = b.placeholder("W", Shape.scalar(), classOf[java.lang.Double])
    val x2 = b.identity(x)

    val result = s.runner()
      .feed(x, Tensor.create(1.0, classOf[java.lang.Double]))
      .fetch(x2).run().get(0)
    assert(result.doubleValue() === 1)

    val result2 = s.runner()
      .feed(x, Tensor.create(2.0, classOf[java.lang.Double]))
      .fetch(x2).run().get(0)
    assert(result2.doubleValue() === 2)
  }


  test("linear regression") {
    val g = new Graph
    val b = new TensorflowGraphBuilder(g)
    val s = new Session(g)

    println(new String(tf.registeredOpList()))


    val x = b.placeholder("x", Shape.scalar(), classOf[java.lang.Double])
    val m = b.variable("m", Shape.scalar(), classOf[java.lang.Double])
    val mInit = b.assign(b.constant("mInit", 3.0), m)
    val yHat = b.mul(x,m)

    s.runner().fetch(mInit).run()
    val result = s.runner()
      .feed(x, Tensor.create(2.0, classOf[java.lang.Double]))
      .fetch(yHat).run().get(0)


    println(result.doubleValue())

  }


}
