package com.microsoft.ml.spark.nn

import java.io._

import breeze.linalg.DenseVector
import com.microsoft.ml.spark.core.test.base.TestBase
import org.apache.commons.io.IOUtils

import scala.collection.immutable
import spray.json._
import spray.json.DefaultJsonProtocol._
import java.io._

import com.microsoft.ml.spark.core.env.StreamUtilities
import com.microsoft.ml.spark.core.test.benchmarks.Benchmarks
import org.apache.commons.math3.stat.descriptive.rank.Percentile


class ConditionalBallTreeTest extends Benchmarks with BallTreeTestBase {
  def naiveSearch(haystack: IndexedSeq[VectorWithExternalId],
                  labels: IndexedSeq[Int],
                  conditioner: Set[Int],
                  needle: DenseVector[Double],
                  k: Int): Seq[BestMatch] = {
    haystack
      .zip(labels)
      .filter(vl => conditioner(vl._2))
      .map(vl => vl._1)
      .map(c => (c, c.features.t * needle))
      .sorted(Ordering.by({ p: (VectorWithExternalId, Double) => (p._2, p._1.id) }).reverse)
      .take(k)
      .map { case (dv, d) => BestMatch(dv.id, d) }
  }

  def assertEquivalent(r1: Seq[BestMatch],
                       r2: Seq[BestMatch],
                       conditioner: Set[Int],
                       labels: IndexedSeq[Int]): Unit = {
    r1.zip(r2).foreach { case (cm, gt) =>
      assert(cm.value === gt.value)
      assert(conditioner(labels(cm.index)))
      assert(conditioner(labels(gt.index)))
    }
  }

  def compareToNaive(haystack: IndexedSeq[VectorWithExternalId],
                     labels: IndexedSeq[Int],
                     conditioner: Set[Int],
                     k: Int,
                     needle: DenseVector[Double]): Unit = {
    val tree = ConditionalBallTree(haystack, labels)
    val conditionalMatches = time("knn", tree.findMaximumInnerProducts(needle, conditioner, k))
    val groundTruth = time("naive", naiveSearch(haystack, labels, conditioner, needle, k))
    assertEquivalent(conditionalMatches, groundTruth, conditioner, labels)
  }

  test("search should be exact") {
    val labels = twoClassLabels(largeUniformData)
    val needle = DenseVector(9.0, 10.0, 11.0)
    val conditioners = Seq(Set(1), Set(2), Set(1, 2))
    val ks = Seq(1, 5, 10, 100)

    for (conditioner <- conditioners; k <- ks) {
      compareToNaive(largeUniformData, labels, conditioner, k, needle)
    }
  }

  test("Balltree with uniform data") {
    val labels = twoClassLabels(uniformData)
    val keys = Seq( //DenseVector(0.0, 0.0, 0.0),
      DenseVector(2.0, 2.0, 20.0),
      DenseVector(9.0, 10.0, 11.0))
    val conditioners = Seq(Set(1), Set(2), Set(1, 2))

    for (key <- keys; conditioner <- conditioners) {
      compareToNaive(uniformData, labels, conditioner, 5, key)
    }
  }

  test("Balltree with random data should be correct") {
    val labels = twoClassLabels(randomData)
    compareToNaive(randomData, labels, Set(1), 5, DenseVector(randomData(3).features.toArray))
  }

  test("Balltree with random data and number of best matches should be correct") {
    val labels = twoClassLabels(randomData)
    val needle = DenseVector(randomData(3).features.toArray)
    compareToNaive(randomData, labels, Set(1), 5, needle)
  }

  test("Balltree should be serializable") {
    val labels = twoClassLabels(uniformData)
    val tree = ConditionalBallTree(uniformData, labels)

    val fos = new FileOutputStream("a.tmp")
    val oos = new ObjectOutputStream(fos)
    oos.writeObject(tree)
    oos.close()
    fos.close()

    val fis = new FileInputStream("a.tmp")
    val ois = new ObjectInputStream(fis)
    val treeDeserialized: ConditionalBallTree[Int] = ois.readObject().asInstanceOf[ConditionalBallTree[Int]]
    println(treeDeserialized.findMaximumInnerProducts(DenseVector(9.0, 10.0, 11.0), Set(1)))
  }

  def mean(xs: Seq[Long]): Double = xs match {
    case Nil => 0.0
    case ys => ys.sum / ys.size.toDouble
  }

  def stddev(xs: Seq[Long], avg: Double): Double = xs match {
    case Nil => 0.0
    case ys => math.sqrt((0.0 /: ys) {
      (a, e) => a + math.pow(e - avg, 2.0)
    } / xs.size)
  }

  def profile[R](block: => R, n: Int = 10): (R, Seq[Double]) = {
    val results = (0 until n).map { i =>
      val t0 = System.nanoTime()
      val result = block
      val t1 = System.nanoTime()
      (result, (t1 - t0)/1e6)
    }
    (results.last._1, results.map(_._2))
  }

  def buildLabelSpecificTrees(data: IndexedSeq[VectorWithExternalId],
                              labels: IndexedSeq[Int],
                              leafNodeSize:Int): Map[Int, BallTree] = {
    labels.toSet.map { label: Int =>
      label -> BallTree(labels.zipWithIndex.filter { case (l, _) => l == label }.map(p => data(p._2)), leafNodeSize)
    }.toMap
  }

  val r = scala.util.Random

  def randomQueryBF(haystack: IndexedSeq[VectorWithExternalId],
                    conditioner: Set[Int],
                    k: Int,
                    labels: IndexedSeq[Int]): Seq[BestMatch] = {
    val key = haystack(r.nextInt(haystack.length))
    haystack
      .zip(labels)
      .filter(vl => conditioner(vl._2))
      .map(vl => vl._1)
      .map(c => (c, c.features.t * key.features))
      .sorted(Ordering.by((_: (VectorWithExternalId, Double))._2).reverse)
      .map(p => BestMatch(p._1.id, p._2))
      .take(k)
  }

  def randomQueryEnsemble(haystack: IndexedSeq[VectorWithExternalId],
                          ensemble: Map[Int, BallTree],
                          conditioner: Set[Int],
                          k: Int): Seq[BestMatch] = {
    val key = haystack(r.nextInt(haystack.length))
    conditioner.flatMap(c =>
      ensemble(c).findMaximumInnerProducts(key.features, k)
    ).toList.sorted(Ordering.by((_: BestMatch).value).reverse).take(k)
  }

  def randomQueryConditional(haystack: IndexedSeq[VectorWithExternalId],
                             ct: ConditionalBallTree[Int],
                             conditioner: Set[Int],
                             k: Int): Seq[BestMatch] = {
    val key = haystack(r.nextInt(haystack.length))
    ct.findMaximumInnerProducts(key.features, conditioner, k)
  }

  def randomQueryFull(haystack: IndexedSeq[VectorWithExternalId],
                      bt: BallTree,
                      k: Int): Seq[BestMatch] = {
    val key = haystack(r.nextInt(haystack.length))
    bt.findMaximumInnerProducts(key.features, k)
  }

  def randomQueryAdaptiveRetry(haystack: IndexedSeq[VectorWithExternalId],
                               bt: BallTree,
                               conditioner: Set[Int],
                               k: Int,
                               labels: IndexedSeq[Int]): Seq[BestMatch] = {
    val key = haystack(r.nextInt(haystack.length))
    var currentK = k
    var matches: Seq[BestMatch] = null
    while (matches == null) {
      if (currentK > bt.points.length) {
        currentK = bt.points.length
      }
      val foundPoints = bt.findMaximumInnerProducts(key.features, currentK)
        .filter(p => conditioner(labels(p.index)))
      if (foundPoints.length < k) {
        currentK = currentK * 2
      } else {
        matches = foundPoints.take(k)
      }
    }
    matches
  }

  /*
  test("profile creating trees by dim") {

    val sizes = Seq(Math.pow(2, 16).toInt)
    val dims = (1 to 11).map(Math.pow(2, _).toInt)
    val nClassesList = Seq(2)
    val allMetrics = for {size<-sizes; dim<-dims; nClasses <- nClassesList} yield {
      println(size, dim, nClasses, Math.log10(size)/Math.log10(2.0))
      val data = randomData(size, dim)
      val labels = randomClassLabels(data, nClasses)
      assert(data.length == size)
      assert(labels.length == size)

      BallTree(data)
      val (treeF, muF, sigmaF) = profileMillis(BallTree(data), 1)
      println(muF)
      val (treeC, muC, sigmaC) = profileMillis(ConditionalBallTree(data, labels), 1)
      println(muC)
      val (treeE, muE, sigmaE) = profileMillis(buildLabelSpecificTrees(data, labels), 1)
      println(muE)

      Map(
        "muC"-> muC,
        "muE"-> muE,
        "muF"-> muF,
        "sigmaC"-> sigmaC,
        "sigmaE"-> sigmaE,
        "sigmaF"-> sigmaF,
        "size" -> size,
        "dim" -> dim,
        "nclasses" -> nClasses
      ).mapValues(_.toString)
    }
    val f = new File(new File(resourcesDirectory, "new_benchmarks"), "treeCreationMetricsByDim.json")
    StreamUtilities.using(new PrintWriter(f)){_.write(allMetrics.toJson.compactPrint)}
  }

  test("profile creating trees by num classes") {

    val sizes = Seq(Math.pow(2, 16).toInt)
    val dims = Seq(512)
    val nClassesList = (1 to 11).map(Math.pow(2, _).toInt)
    val allMetrics = for {size<-sizes; dim<-dims; nClasses <- nClassesList} yield {
      println(size, dim, nClasses, Math.log10(size)/Math.log10(2.0))
      val data = randomData(size, dim)
      val labels = randomClassLabels(data, nClasses)
      assert(data.length == size)
      assert(labels.length == size)

      BallTree(data)
      val (treeF, muF, sigmaF) = profileMillis(BallTree(data), 1)
      println(muF)
      val (treeC, muC, sigmaC) = profileMillis(ConditionalBallTree(data, labels), 1)
      println(muC)
      val (treeE, muE, sigmaE) = profileMillis(buildLabelSpecificTrees(data, labels), 1)
      println(muE)

      Map(
        "muC"-> muC,
        "muE"-> muE,
        "muF"-> muF,
        "sigmaC"-> sigmaC,
        "sigmaE"-> sigmaE,
        "sigmaF"-> sigmaF,
        "size" -> size,
        "dim" -> dim,
        "nclasses" -> nClasses
      ).mapValues(_.toString)
    }
    val f = new File(new File(resourcesDirectory, "new_benchmarks"), "treeCreationMetricsByClasses.json")
    StreamUtilities.using(new PrintWriter(f)){_.write(allMetrics.toJson.compactPrint)}
  }*/

  import session.implicits._

  def getSummaryStats(times: Seq[Double], name: String): Map[String, Double] = {
    val arr = times.toArray
    Map(
      "min" -> arr.min,
      "10" -> new Percentile().evaluate(arr, 10),
      "25" -> new Percentile().evaluate(arr, 25),
      "50" -> new Percentile().evaluate(arr, 50),
      "75" -> new Percentile().evaluate(arr, 75),
      "90" -> new Percentile().evaluate(arr, 90),
      "max" -> arr.max
    ).map(kv => (name + "_" + kv._1, kv._2))
  }

  test("query trees by num classes") {
    val size = 100000 //Math.pow(2, 17).toInt
    val dim =  4096
    val nClasses = 1024
    val leafSize = 10
    val subsetSizes = Seq(1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024)
    val ks = Seq(1, 10)

    val data = randomData(size, dim)
    val labels = randomClassLabels(data, nClasses)
    assert(data.length == size)
    assert(labels.length == size)

    //val (treeF, _) = profile(BallTree(data, leafSize), 1)
    println("here1")
    val (treeC, tc) = profile(ConditionalBallTree(data, labels, leafSize), 1)
    println(tc)
    val (treeE, te) = profile(buildLabelSpecificTrees(data, labels, leafSize), 1)
    println(te)

    val allMetrics = for {subsetSize <- subsetSizes;
                          k <- ks} yield {
      val conditioner = (0 until subsetSize).toSet
      println(subsetSize, k)

      //System.gc()
      //val (resultBF, timesBF) = profile(randomQueryBF(data, conditioner, k, labels), 10)
      //println("here1")
      //System.gc()
      //val (resultF, timesF) = profile(randomQueryFull(data, treeF, k), 20)
      //println("here2")
      //System.gc()
      //val (resultAR, timesAR) = profile(randomQueryAdaptiveRetry(data, treeF, conditioner, k, labels), 30)
      //println("here3")
      System.gc()
      val (resultC, timesC) = profile(randomQueryConditional(data, treeC, conditioner, k), 20)
      println("here4")
      System.gc()
      val (resultE, timesE) = profile(randomQueryEnsemble(data, treeE, conditioner, k), 20)
      println("here5")

      val allStats = (
        //getSummaryStats(timesBF, "BF") ++
        //getSummaryStats(timesF, "F") ++
        getSummaryStats(timesC, "C") ++
        //getSummaryStats(timesAR, "AR") ++
        getSummaryStats(timesE, "E") ++
          Map(
        "subsetSize"->subsetSize,
        "k" -> k,
        "size" -> size,
        "dim" -> dim,
        "nClasses" -> nClasses
        )).mapValues(_.toString)
      println(allStats)
      allStats
    }

    val dir = new File(resourcesDirectory, "new_benchmarks")
    if (!dir.exists()){dir.mkdirs()}
    val f = new File(dir, "queryTimesBySubsetSize.json")
    StreamUtilities.using(new PrintWriter(f)){_.write(allMetrics.toJson.compactPrint)}
  }

  /*
  test("profile creating trees") {
    val sizes = (1 to 23).map(Math.pow(2, _).toInt)
    val dims = Seq(3)
    val nClassesList = Seq(2)
    val allMetrics = for {size<-sizes; dim<-dims; nClasses <- nClassesList} yield {
      println(size, dim, nClasses, Math.log10(size)/Math.log10(2.0))
      val data = randomData(size, dim)
      val labels = randomClassLabels(data, nClasses)
      assert(data.length == size)
      assert(labels.length == size)

      val (treeF, muF, sigmaF) = profileMillis(BallTree(data), 1)
      println(muF)
      val (treeC, muC, sigmaC) = profileMillis(ConditionalBallTree(data, labels), 1)
      println(muC)
      val (treeE, muE, sigmaE) = profileMillis(buildLabelSpecificTrees(data, labels), 1)
      println(muE)

      Map(
        "muC"-> muC,
        "muE"-> muE,
        "muF"-> muF,
        "sigmaC"-> sigmaC,
        "sigmaE"-> sigmaE,
        "sigmaF"-> sigmaF,
        "size" -> size,
        "dim" -> dim,
        "nclasses" -> nClasses
      ).mapValues(_.toString)
    }

    val f = new File(new File(resourcesDirectory, "new_benchmarks"), "treeCreationMetricsBySize.json")
    StreamUtilities.using(new PrintWriter(f)){_.write(allMetrics.toJson.compactPrint)}

  }

  test("compare to baselines") {
    val labels
    val (treeC, muC, sigmaC) = profileMillis(ConditionalBallTree(largeUniformData, labels), 3)
    val (treeF, muF, sigmaF) = profileMillis(BallTree(largeUniformData), 3)
    val (treeE, muE, sigmaE) = profileMillis(buildLabelSpecificTrees(largeUniformData, labels), 3)

    val key = largeUniformData(0)
    val conditioner = Set("A")
    val k = 5
    val (resultsC, muQC, sigmaQC) =  profileMillis(queryConditional(treeC, conditioner, key, k),  50)
    val (resultsE, muQE, sigmaQE) =  profileMillis(queryEnsemble(treeE, conditioner, key, k), 50)
    val (resultsF, muQF, sigmaQF) =  profileMillis(queryFull(treeF, conditioner, key, k, labels),  50)
    val (resultsBF, muQBF, sigmaQBF) =  profileMillis(queryBF(largeUniformData, conditioner, key, k, labels),  50)

    val metrics = Map(
      "muC"-> muC,
      "muE"-> muE,
      "muF"-> muF,
      "sigmaC"-> sigmaC,
      "sigmaE"-> sigmaE,
      "sigmaF"-> sigmaF,
      "muQC"-> muQC,
      "muQE"-> muQE,
      "muQF"-> muQF,
      "muQBF"-> muQBF,
      "sigmaQC"-> sigmaQC,
      "sigmaQE"-> sigmaQE,
      "sigmaQF"-> sigmaQF,
      "sigmaQBF"-> sigmaQBF,
      "dataset" -> "largeUniform",
      "nrows" -> s"${largeUniformData.length}",
      "nclasses" -> "2"
    )

    assertEquivalent(resultsC, resultsBF, conditioner, labels)
    assertEquivalent(resultsE, resultsBF, conditioner, labels)
    assertEquivalent(resultsF, resultsBF, conditioner, labels)
  }*/
}