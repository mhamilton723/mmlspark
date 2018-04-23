// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.ml.spark

import com.microsoft.ml.spark.schema.SparkBindings
import org.apache.http.client.methods.HttpPost
import org.apache.spark.sql.Row

case class Person(name: String, age: Option[Int])

object Person extends SparkBindings[Person]

class BindingSuite extends TestBase {

  test("encoding with missing vals") {
    val p1 = Person("foo", Some(1))
    val r1 = Person.toRow(p1)
    val pp1 = Person.fromRow(r1)
    assert(p1 === pp1)

    val p2 = Person("foo2", None)
    val r2 = Person.toRow(p2)
    val pp2 = Person.fromRow(r2)
    assert(p2 === pp2)
  }

  test("edata") {
    import session.implicits._
    val barr: Array[Byte] = Array(1)
    val e = EntityData(barr, None, 1L, None, false, false, false)
    val r = EntityData.toRow(e)
    val e2 = EntityData.fromRow(r)
    val e3 = EntityData.fromRow(Row(barr, null, 1L, null, false, false, false))
    println(e2)
    println(e3)

    val post = new HttpPost()
    val req = new HTTPRequestData(post)
    val row = HTTPRequestData.toRow(req)
    val req2 = HTTPRequestData.fromRow(row)
    println(req2)
  }

}
