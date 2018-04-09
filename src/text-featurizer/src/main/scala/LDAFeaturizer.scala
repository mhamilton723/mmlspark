// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.ml.spark

import java.util.NoSuchElementException

import com.microsoft.ml.spark.schema.DatasetExtensions
import org.apache.spark.ml.{Pipeline, _}
import org.apache.spark.ml.attribute.AttributeGroup
import org.apache.spark.ml.clustering.LDA
import org.apache.spark.ml.feature._
import org.apache.spark.ml.param._
import org.apache.spark.ml.util._
import org.apache.spark.sql.types._
import org.apache.spark.sql.{DataFrame, Dataset}

import scala.reflect.runtime.universe.{TypeTag, typeTag}

trait LDAFeaturizerParams extends Wrappable with DefaultParamsWritable {

  /** Tokenize the input when set to true
    * @group param
    */
  val useTokenizer = BooleanParam(this, "useTokenizer", "Whether to tokenize the input")

  /** @group getParam */
  final def getUseTokenizer: Boolean = $(useTokenizer)

  /** Indicates whether the regex splits on gaps (true) or matches tokens (false)
    * @group param
    */
  val tokenizerGaps = BooleanParam(
    this,
    "tokenizerGaps",
    "Indicates whether regex splits on gaps (true) or matches tokens (false)."
  )

  /** @group getParam */
  final def getTokenizerGaps: Boolean = $(tokenizerGaps)

  /** Minumum token length; must be 0 or greater.
    * @group param
    */
  val minTokenLength = IntParam(this, "minTokenLength", "Minimum token length, >= 0.")

  /** @group getParam */
  final def getMinTokenLength: Int = $(minTokenLength)

  /** Regex pattern used to match delimiters if gaps (true) or tokens (false)
    * @group param
    */
  val tokenizerPattern = StringParam(
    this,
    "tokenizerPattern",
    "Regex pattern used to match delimiters if gaps is true or tokens if gaps is false.")

  /** @group getParam */
  final def getTokenizerPattern: String = $(tokenizerPattern)

  /** Indicates whether to convert all characters to lowercase before tokenizing.
    * @group param
    */
  val toLowercase = BooleanParam(
    this,
    "toLowercase",
    "Indicates whether to convert all characters to lowercase before tokenizing.")

  /** @group getParam */
  final def getToLowercase: Boolean = $(toLowercase)

  /** Indicates whether to remove stop words from tokenized data.
    * @group param
    */
  val useStopWordsRemover = BooleanParam(this,
    "useStopWordsRemover",
    "Whether to remove stop words from tokenized data")

  /** @group getParam */
  final def getUseStopWordsRemover: Boolean = $(useStopWordsRemover)

  /** Indicates whether a case sensitive comparison is performed on stop words.
    * @group param
    */
  val caseSensitiveStopWords = BooleanParam(
    this,
    "caseSensitiveStopWords",
    " Whether to do a case sensitive comparison over the stop words")

  /** @group getParam */
  final def getCaseSensitiveStopWords: Boolean = $(caseSensitiveStopWords)

  /** Specify the language to use for stop word removal. The Use the custom setting when using the
    * stopWords input
    * @group param
    */
  val defaultStopWordLanguage = StringParam(this,
    "defaultStopWordLanguage",
    "Which language to use for the stop word remover," +
      " set this to custom to use the stopWords input")

  /** @group getParam */
  final def getDefaultStopWordLanguage: String = $(defaultStopWordLanguage)

  /** The words to be filtered out. This is a comma separated list of words, encoded as a single string.
    * For example, "a, the, and"
    */
  val stopWords = StringParam(this, "stopWords", "The words to be filtered out.")

  /** @group getParam */
  final def getStopWords: String = $(stopWords)

  /** Enumerate N grams when set
    * @group param
    */
  val useNGram = BooleanParam(this, "useNGram", "Whether to enumerate N grams")

  /** @group getParam */
  final def getUseNGram: Boolean = $(useNGram)

  /** The size of the Ngrams
    * @group param
    */
  val nGramLengths = new ArrayParam(this, "nGramLengths", "The size of the Ngrams")

  /** @group getParam */
  final def getNGramLengths: Array[Int] = $(nGramLengths).asInstanceOf[Array[Int]]

  /** @group getParam */
  final def getNGramLength: Int = {
    val lengths = getNGramLengths
    assert(lengths.lengthCompare(1) == 0, "more than 1 ngram length is provided")
    lengths.head
  }

  /** @group getParam */
  final def getNGramRange: (Int, Int) = {
    val lengths = getNGramLengths.sorted
    assert(lengths.last - lengths.head +1 == lengths.length,
      s"Ngram Lengths: $getNGramLengths do not conform to a range pattern")
    (lengths.head, lengths.last)
  }

  /** All nonnegative word counts are set to 1 when set to true
    * @group param
    */
  val binary = BooleanParam(
    this,
    "binary",
    "If true, all nonegative word counts are set to 1")

  /** @group getParam */
  final def getBinary: Boolean = $(binary)

  /** Set the number of features to hash each document to
    * @group param
    */
  val numFeatures = IntParam(
    this,
    "numFeatures",
    "Set the number of features to hash each document to")

  /** @group getParam */
  final def getNumFeatures: Int = $(numFeatures)

  /** Scale the Term Frequencies by IDF when set to true
    * @group param
    */
  val useIDF = BooleanParam(
    this,
    "useIDF",
    "Whether to scale the Term Frequencies by IDF")

  /** @group getParam */
  final def getUseIDF: Boolean = $(useIDF)

  /** Minimum number of documents in which a term should appear.
    * @group param
    */
  val minDocFreq = IntParam(
    this,
    "minDocFreq",
    "The minimum number of documents in which a term should appear.")

  /** @group getParam */
  final def getMinDocFreq: Int = $(minDocFreq)

}

object LDAFeaturizer extends DefaultParamsReadable[LDAFeaturizer]

/** Featurize text.
  *
  * @param uid The id of the module
  */
@InternalWrapper
class LDAFeaturizer(override val uid: String)
  extends Estimator[LDAFeaturizerModel]
    with LDAFeaturizerParams with HasInputCol with HasOutputCol {
  def this() = this(Identifiable.randomUID("LDAFeaturizer"))

  setDefault(outputCol, uid + "_output")

  def setUseTokenizer(value: Boolean): this.type = set(useTokenizer, value)

  setDefault(useTokenizer -> true)

  /** @group setParam */
  def setTokenizerGaps(value: Boolean): this.type = set(tokenizerGaps, value)

  setDefault(tokenizerGaps -> true)

  /** @group setParam */
  def setMinTokenLength(value: Int): this.type = set(minTokenLength, value)

  setDefault(minTokenLength -> 0)

  /** @group setParam */
  def setTokenizerPattern(value: String): this.type =
    set(tokenizerPattern, value)

  setDefault(tokenizerPattern -> "\\s+")

  /** @group setParam */
  def setToLowercase(value: Boolean): this.type = set(toLowercase, value)

  setDefault(toLowercase -> true)

  /** @group setParam */
  def setUseStopWordsRemover(value: Boolean): this.type =
    set(useStopWordsRemover, value)

  setDefault(useStopWordsRemover -> false)

  /** @group setParam */
  def setCaseSensitiveStopWords(value: Boolean): this.type =
    set(caseSensitiveStopWords, value)

  setDefault(caseSensitiveStopWords -> false)

  /** @group setParam */
  def setDefaultStopWordLanguage(value: String): this.type =
    set(defaultStopWordLanguage, value)

  setDefault(defaultStopWordLanguage -> "english")

  /** @group setParam */
  def setStopWords(value: String): this.type = set(stopWords, value)

  /** @group setParam */
  def setUseNGram(value: Boolean): this.type = set(useNGram, value)

  /** @group setParam */
  def setNGramLength(value: Int): this.type = set(nGramLengths, Array(value))

  /** @group setParam */
  def setNGramLengths(values: Array[Int]): this.type = set(nGramLengths, values)

  /** @group setParam */
  def setNGramRange(min: Int, max: Int): this.type = set(nGramLengths, (min to max).toArray)

  /** @group setParam */
  def setNGramRange(range: (Int, Int)): this.type = set(nGramLengths, (range._1 to range._2).toArray)

  setDefault(useNGram -> false, nGramLengths -> Array(2))

  /** @group setParam */
  def setBinary(value: Boolean): this.type = set(binary, value)

  /** @group setParam */
  def setNumFeatures(value: Int): this.type = set(numFeatures, value)

  setDefault(numFeatures -> (1 << 18), binary -> false)

  /** @group setParam */
  def setUseIDF(value: Boolean): this.type = set(useIDF, value)

  /** @group setParam */
  def setMinDocFreq(value: Int): this.type = set(minDocFreq, value)

  setDefault(useIDF -> true, minDocFreq -> 1)

  private def setParamInternal[M <: PipelineStage, T](model: M,
                                                      name: String,
                                                      value: T) = {
    model.set(model.getParam(name), value)
  }

  private def getParamInternal[M <: PipelineStage, T](model: M, name: String) = {
    model.getOrDefault(model.getParam(name))
  }

  private def makePipe(schema: StructType) = {
    val featuresCol = DatasetExtensions.findUnusedColumnName("features", schema)
    val feat = new TextFeaturizer()
      .setInputCol(getInputCol)
      .setOutputCol(featuresCol)
    val lda = new LDA()
      .setFeaturesCol(feat.getOutputCol)
      .setTopicDistributionCol(getTopicDistributionColumn)

    val dc = new DropColumns().setCol(feat.getOutputCol)
     new Pipeline().setStages(Array(feat,lda,dc))
  }

  override def fit(dataset: Dataset[_]): LDAFeaturizerModel = {
    new LDAFeaturizerModel(uid, makePipe(dataset.schema).fit(dataset)).setParent(this)
  }

  override def copy(extra: ParamMap): Estimator[LDAFeaturizerModel] =
    defaultCopy(extra)

  override def transformSchema(schema: StructType): StructType = {
    makePipe(schema).transformSchema(schema)
  }

}

class LDAFeaturizerModel(val uid: String,
                          fitPipeline: PipelineModel)
  extends Model[LDAFeaturizerModel] with ConstructorWritable[LDAFeaturizerModel] {

  override def copy(extra: ParamMap): LDAFeaturizerModel = defaultCopy(extra)

  override def transform(dataset: Dataset[_]): DataFrame = {
    fitPipeline.transform(dataset)
  }

  override def transformSchema(schema: StructType): StructType =
    fitPipeline.transformSchema(schema)

  override val ttag: TypeTag[LDAFeaturizerModel] = typeTag[LDAFeaturizerModel]
  override def objectsToSave: List[AnyRef] = List(uid, fitPipeline)
}

object LDAFeaturizerModel extends ConstructorReadable[LDAFeaturizerModel]
