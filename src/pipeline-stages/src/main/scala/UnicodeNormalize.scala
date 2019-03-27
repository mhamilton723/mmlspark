// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.ml.spark

import org.apache.spark.ml.Transformer
import org.apache.spark.ml.param.{BooleanParam, Param, ParamMap}
import org.apache.spark.ml.util.{ComplexParamsReadable, ComplexParamsWritable, Identifiable}
import org.apache.spark.sql.{DataFrame, Dataset}
import org.apache.spark.sql.functions.udf
import java.text.Normalizer

import org.apache.spark.sql.types.{StringType, StructField, StructType}

object UnicodeNormalize extends ComplexParamsReadable[UnicodeNormalize]

/** <code>UnicodeNormalize</code> takes a dataframe and normalizes the unicode representation.
  */
class UnicodeNormalize(val uid: String) extends Transformer
  with HasInputCol with HasOutputCol with Wrappable with ComplexParamsWritable {
  def this() = this(Identifiable.randomUID("UnicodeNormalize"))

  val form = new Param[String](this, "form", "Unicode normalization form: NFC, NFD, NFKC, NFKD")

  /** @group getParam */
  def getForm: String = get(form).getOrElse("NFKD")

  /** @group setParam */
  def setForm(value: String): this.type = {
    // check input value
    Normalizer.Form.valueOf(getForm)

    set("form", value)
  }

  val lower = new BooleanParam(this, "lower", "Lowercase text")

  /** @group getParam */
  def getLower: Boolean = get(lower).getOrElse(true)

  /** @group setParam */
  def setLower(value: Boolean): this.type = set("lower", value)

  /** @param dataset - The input dataset, to be transformed
    * @return The DataFrame that results from column selection
    */
  override def transform(dataset: Dataset[_]): DataFrame = {
    val inputIndex = dataset.columns.indexOf(getInputCol)

    require(inputIndex != -1, s"Input column $getInputCol does not exist")

<<<<<<< HEAD
    val normalizeFunc = (value: String) =>
      if (value == null) null
      else Normalizer.normalize(value, Normalizer.Form.valueOf(getForm))

    val f = if (getLower)
      (value: String) => if (value == null) null else normalizeFunc(value.toLowerCase)
=======
    val normalizeFunc = (value: String) => Normalizer.normalize(value, Normalizer.Form.valueOf(getForm))

    val f = if (getLower)
      (value: String) => normalizeFunc(value.toLowerCase)
>>>>>>> 0b84a230d1556ced87be9139dd798237711c1158
    else
      normalizeFunc

    val textMapper = udf(f)

    dataset.withColumn(getOutputCol, textMapper(dataset(getInputCol)).as(getOutputCol))
  }

  def transformSchema(schema: StructType): StructType = {
    schema.add(StructField(getOutputCol, StringType))
  }

  def copy(extra: ParamMap): UnicodeNormalize = defaultCopy(extra)

}
