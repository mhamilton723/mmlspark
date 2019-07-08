// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.ml.spark.vw

import com.microsoft.ml.spark.core.env.InternalWrapper
import com.microsoft.ml.spark.core.serialize.{ConstructorReadable, ConstructorWritable}
import org.apache.spark.ml.param._
import org.apache.spark.ml.util._
import org.apache.spark.ml.classification.{ProbabilisticClassificationModel, ProbabilisticClassifier}
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.ml.param.shared.HasProbabilityCol
import org.apache.spark.sql._
import org.apache.spark.sql.functions.{col, udf}

import scala.reflect.runtime.universe.{TypeTag, typeTag}
import scala.math.exp

object VowpalWabbitClassifier extends DefaultParamsReadable[VowpalWabbitClassifier]

@InternalWrapper
class VowpalWabbitClassifier(override val uid: String)
  extends ProbabilisticClassifier[Row, VowpalWabbitClassifier, VowpalWabbitClassificationModel]
  with VowpalWabbitBase
{
  def this() = this(Identifiable.randomUID("VowpalWabbitClassifier"))

  override protected def train(dataset: Dataset[_]): VowpalWabbitClassificationModel = {

    val binaryModel = trainInternal(dataset)

    new VowpalWabbitClassificationModel(uid, binaryModel, getLabelCol, getFeaturesCol, getAdditionalFeatures,
      getPredictionCol, getProbabilityCol, getRawPredictionCol)
  }

  override def copy(extra: ParamMap): VowpalWabbitClassifier = defaultCopy(extra)
}

// Preparation for multi-class learning, though it no fun as numClasses is spread around multiple reductions
@InternalWrapper
class VowpalWabbitClassificationModel(
    override val uid: String, val model: Array[Byte], labelColName: String,
    featuresColName: String, additionalFeaturesName: Array[String],
    predictionColName: String, probabilityColName: String, rawPredictionColName: String)
  extends ProbabilisticClassificationModel[Row, VowpalWabbitClassificationModel]
    with VowpalWabbitBaseModel with HasProbabilityCol // TODO: HasThresholds
    with ConstructorWritable[VowpalWabbitClassificationModel]
{
  def numClasses: Int = 2

  set(labelCol, labelColName)
  set(featuresCol, featuresColName)
  set(additionalFeatures, additionalFeaturesName)
  set(predictionCol, predictionColName)
  set(probabilityCol, probabilityColName)
  set(rawPredictionCol, rawPredictionColName)

  override def transform(dataset: Dataset[_]): DataFrame = {
    val df = transformImplInternal(dataset)

    // which mode one wants to use depends a bit on how this should be deployed
    // 1. if you stay in spark w/o link=logistic is probably more convenient as it also returns the raw prediction
    // 2. if you want to export the model *and* get probabilities at scoring term w/ link=logistic is preferable

    // convert raw prediction to probability (if needed)
    val probabilityUdf = if (vwArgs.getArgs.contains("--link=logistic"))
      udf { (pred: Double) => Vectors.dense(Array(1 - pred, pred)) }
    else
      udf { (pred: Double) => {
        val prob = 1.0 / (1.0 + exp(-pred))
        Vectors.dense(Array(1 - prob, prob))
      } }

    val df2 = df.withColumn($(probabilityCol), probabilityUdf(col($(rawPredictionCol))))

    // convert probability to prediction
    val probability2predictionUdf = udf(probability2prediction _)
    df2.withColumn($(predictionCol), probability2predictionUdf(col($(probabilityCol))))
  }

  override def copy(extra: ParamMap): VowpalWabbitClassificationModel =
    new VowpalWabbitClassificationModel(uid, model, getLabelCol, getFeaturesCol, getAdditionalFeatures,
      getPredictionCol, getProbabilityCol, getRawPredictionCol)

  protected override def predictRaw(features: Row): Vector = {
    throw new NotImplementedError("Not implemented")
  }

  protected override def raw2probabilityInPlace(rawPrediction: Vector): Vector= {
    throw new NotImplementedError("Not implemented")
  }

  override val ttag: TypeTag[VowpalWabbitClassificationModel] =
    typeTag[VowpalWabbitClassificationModel]

  override def objectsToSave: List[Any] =
    List(uid, model, getLabelCol, getFeaturesCol, getAdditionalFeatures, getPredictionCol,
      getProbabilityCol, getRawPredictionCol)
}

object VowpalWabbitClassificationModel extends ConstructorReadable[VowpalWabbitClassificationModel]
