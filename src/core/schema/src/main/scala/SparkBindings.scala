// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.ml.spark.schema

import org.apache.spark.sql.Row
import org.apache.spark.sql.catalyst.encoders.{ExpressionEncoder, RowEncoder}
import org.apache.spark.sql.types.StructType

import scala.reflect.runtime.universe.TypeTag

abstract class SparkBindings[T: TypeTag] {
  lazy val enc: ExpressionEncoder[T] = ExpressionEncoder[T]().resolveAndBind()
  lazy val rowEnc: ExpressionEncoder[Row] = RowEncoder(enc.schema).resolveAndBind()
  lazy val schema: StructType = enc.schema

  def fromRow(r: Row): T = {
    val ir = rowEnc.toRow(r)
    enc.fromRow(ir)
  }

  def fromRowOpt(r: Row): Option[T] = {
    try {
      val ir = rowEnc.toRow(r)
      Some(enc.fromRow(ir))
    } catch {
      case _: Exception =>
        None //TODO figure out why this is needed for certain classes
    }
  }

  def toRow(v: T): Row = {
    rowEnc.fromRow(enc.toRow(v))
  }

}
