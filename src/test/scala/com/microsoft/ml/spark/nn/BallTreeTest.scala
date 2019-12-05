package com.microsoft.ml.spark.nn

import java.io._

import breeze.linalg.DenseVector
import com.microsoft.ml.spark.core.test.base.TestBase
import scala.collection.immutable

trait BallTreeTestBase extends TestBase {
  def time[R](identifier: String = "", block: => R): R = {
    val t0 = System.currentTimeMillis()
    val result = block // call-by-name
    val t1 = System.currentTimeMillis()
    println(s"Elapsed time for ${identifier}: " + (t1 - t0) + "ms")
    result
  }

  lazy val uniformData: immutable.IndexedSeq[VectorWithExternalId] = {
    {
      for (x <- 1 to 10; y <- 1 to 10; z <- 1 to 10)
        yield DenseVector((x * 2).toDouble, (y * 2).toDouble, (z * 2).toDouble)
      }.zipWithIndex.map { case (dv, i) => VectorWithExternalId(i, dv) }
  }

  lazy val largeUniformData: immutable.IndexedSeq[VectorWithExternalId] = {
    {
      for (x <- 1 to 100; y <- 1 to 100; z <- 1 to 100)
        yield DenseVector((x * 2).toDouble, (y * 2).toDouble, (z * 2).toDouble)
      }.zipWithIndex.map { case (dv, i) => VectorWithExternalId(i, dv) }
  }

  lazy val veryLargeUniformData: immutable.IndexedSeq[VectorWithExternalId] = {
    {
      for (x <- 1 to 200; y <- 1 to 100; z <- 1 to 100)
        yield DenseVector((x * 2).toDouble, (y * 2).toDouble, (z * 2).toDouble)
      }.zipWithIndex.map { case (dv, i) => VectorWithExternalId(i, dv) }
  }

  lazy val randomData: immutable.IndexedSeq[VectorWithExternalId] = {
    scala.util.Random.setSeed(10)
    def random(n: Int): immutable.IndexedSeq[Double] = (1 to n).map(_ => (scala.util.Random.nextDouble - 0.5) * 2)
    (1 to 100000).map(_ => DenseVector(random(3).toArray))
      .zipWithIndex.map { case (dv, i) => VectorWithExternalId(i, dv)}
  }

  def twoClassLabels(data: IndexedSeq[_]): IndexedSeq[Int] =
    IndexedSeq.fill(data.length / 2)(1) ++ IndexedSeq.fill(data.length - data.length / 2)(2)

  def randomClassLabels(data: IndexedSeq[_], nClasses: Int): IndexedSeq[Int] = {
    val r = scala.util.Random
    data.map(_ => r.nextInt(nClasses))
  }

  def randomData(size: Int, dim: Int): IndexedSeq[VectorWithExternalId] = {
    scala.util.Random.setSeed(10)

    IndexedSeq.fill(size){
      DenseVector.fill(size){(scala.util.Random.nextDouble - 0.5) * 2}
    }.zipWithIndex.map { case (dv, i) => VectorWithExternalId(i, dv)}
  }

}

class BallTreeTest extends BallTreeTestBase {
  def naiveSearch(haystack: IndexedSeq[VectorWithExternalId],
                  needle: DenseVector[Double]): IndexedSeq[(VectorWithExternalId, Double)] = {
    haystack.map(c => (c, c.features.t * needle))
      .sorted(Ordering.by({ p: (VectorWithExternalId, Double) => p._2 }).reverse)
  }

  def compareToGroundTruth(tree: BallTree,
                           haystack: IndexedSeq[VectorWithExternalId],
                           needle: DenseVector[Double],
                           k: Int = 1): (immutable.IndexedSeq[(Int, Double)], Seq[BestMatch]) = {
    val nearest = tree.findMaximumInnerProducts(needle, k)
    val groundTruth = haystack.indices.map { point => (point, haystack(point).features.t * needle) }
    val better = groundTruth.filter(n => n._2 > nearest.head.value).sorted
    (better, nearest)
  }

  def test(haystack: IndexedSeq[VectorWithExternalId],
           needles: DenseVector[Double]*): BestMatch = {
    println()
    val tree = BallTree(haystack)
    val needle = needles(0)
    val nearest = time("knn", tree.findMaximumInnerProducts(needle).head)

    // Brute force proof
    val groundTruth = time("naive", haystack.map { point => point.features.t * needle })
    val better = groundTruth.filter(n => n > nearest.value)
      .sorted
    assert(better.isEmpty)
    nearest
  }

  test("search should be exact") {
    val needle = DenseVector(9.0, 10.0, 11.0)
    val nearest = test(largeUniformData, needle)
    assert(largeUniformData(nearest.index) === naiveSearch(largeUniformData, needle).head._1)

    val tree = BallTree(largeUniformData)
    val nearest2 = time("knn", tree.findMaximumInnerProducts(needle).head)
    val groundTruth2 = time("naive", largeUniformData.map { point => point.features.t * needle }.min)

    val nearest3 = time("knn", tree.findMaximumInnerProducts(needle).head)
    val groundTruth3 = time("naive", largeUniformData.map { point => point.features.t * needle }.min)

    println(largeUniformData.length)
    //      DenseVector(0.0, 0.0, 0.0), DenseVector(2.0, 2.0, 20.0),
  }

  test("Balltree with uniform data") {
    val tree = BallTree(uniformData)
    Seq(DenseVector(0.0, 0.0, 0.0),
      DenseVector(2.0, 2.0, 20.0),
      DenseVector(9.0, 10.0, 11.0)).foreach { vec =>
      val (better, _) = compareToGroundTruth(tree, uniformData, vec)
      assert(better.isEmpty)
    }
  }

  test("Balltree with random data should be correct") {
    val tree = BallTree(randomData)
    val (better, _) = compareToGroundTruth(tree, randomData, DenseVector(randomData(3).features.toArray))
    assert(better.isEmpty)
  }

  test("Balltree with random data and number of best matches should be correct") {
    val needle = DenseVector(randomData(3).features.toArray)
    val tree = BallTree(randomData)
    val (better, nearest) = compareToGroundTruth(tree, randomData, needle, 5)
    assert(better.isEmpty)
    assert(nearest.length == 5)
  }

  test("Balltree should be serializable") {
    val tree = BallTree(uniformData)
    val fos = new FileOutputStream("a.tmp")
    val oos = new ObjectOutputStream(fos)
    oos.writeObject(tree)
    oos.close()
    fos.close()

    val fis = new FileInputStream("a.tmp")
    val ois = new ObjectInputStream(fis)
    val treeDeserialized: BallTree = ois.readObject().asInstanceOf[BallTree]
    println(treeDeserialized.findMaximumInnerProducts(DenseVector(9.0, 10.0, 11.0)))
  }
}