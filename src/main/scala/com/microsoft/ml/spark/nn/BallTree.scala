package com.microsoft.ml.spark.nn

import java.io.Serializable

import breeze.linalg.functions.euclideanDistance
import breeze.linalg.{DenseVector, norm}
import breeze.linalg._


private case class Query(point: DenseVector[Double],
                         normOfQueryPoint: Double,
                         bestMatches: BoundedPriorityQueue[BestMatch],
                         statistics: Statistics) extends Serializable {

  def this(point: DenseVector[Double], bestK: Int) = this(point, norm(point), Query.createQueue(bestK), Statistics())

  override def toString: String = {
    s"Query with point ${point}} \n " +
      s"and bestMatches of size of ${bestMatches.size} (bestMatch example: ${bestMatches.take(1)}})"
  }
}

private object Query {
  def createQueue(k: Int): BoundedPriorityQueue[BestMatch] = {
    val bestMatches = new BoundedPriorityQueue[BestMatch](k)(Ordering.by(_.value))
    bestMatches += BestMatch(-1, Double.NegativeInfinity)
  }
}

private final case class Statistics(var pointsInTree: Int = 1,
                                    var innerProductEvaluations: Int = 0,
                                    var subTreeIgnores: Int = 0,
                                    var subtreesVisited: Int = 0,
                                    var boundEvaluations: Int = 0) extends Serializable {
  override def toString: String = {
    s"innerProductEvaluations: ${innerProductEvaluations}, \n" +
      s"subTreeIgnores: ${subTreeIgnores}, \n" +
      s"subtreesVisited: ${subtreesVisited}, \n" +
      s"boundEvaluations: ${boundEvaluations}, \n" +
      s"innerProductEvaluationsRelativeToNaiveSearch: ${innerProductEvaluations.toDouble / pointsInTree * 100} %"
  }
}

object BenchmarkingUtils {
  def dot2(v1: DenseVector[Double], v2: DenseVector[Double]): Double = {
    counts+=1
    v1 dot v2
  }
  var counts: Int = 0
}
import BenchmarkingUtils._

trait BallTreeBase {

  val points: IndexedSeq[VectorWithExternalId]
  val leafSize: Int
  val epsilon = 0.000001
  val splitSubset = 100

  //using java version of Random() cause the scala version is only serializable since scala version 2.11
  val randomIntGenerator = new java.util.Random()
  val dim: Int = points(0).features.length
  val pointIdx: Range = points.indices

  private def mean(pointIdx: Seq[Int]): DenseVector[Double] = {
    (1.0 / pointIdx.length) * pointIdx.map(points(_).features).reduce(_ + _)
  }

  private def radius(pointIdx: Seq[Int], point: DenseVector[Double]): Double = {
    pointIdx.map { idx =>
      euclideanDistance(points(idx).features, point)
    }.max
  }

  protected def upperBoundMaximumInnerProduct(query: Query, node: Node): Double = {
    query.statistics.boundEvaluations += 1
    (dot2(query.point, node.ball.mu)) + (node.ball.radius * query.normOfQueryPoint)
  }

  private def makeBallSplit(pointIdx: Seq[Int]): (Int, Int) = {
    //finding two points in Set that have largest distance
    val randPoint = points(pointIdx(randomIntGenerator.nextInt(pointIdx.length))).features
    val randSubset = pointIdx

    //TODO: Check if not using squared euclidean distance is ok
    val pivotPoint1: Int = randSubset.map { idx: Int => {
      val ed = euclideanDistance(randPoint, points(idx).features)
      (idx, ed * ed)
    }
    }.maxBy(_._2)._1
    val pivotPoint2: Int = randSubset.map { idx: Int => {
      val ed = euclideanDistance(points(pivotPoint1).features, points(idx).features)
      (idx, ed * ed)
    }
    }.maxBy(_._2)._1
    (pivotPoint1, pivotPoint2)
  }

  private def divideSet(pointIdx: Seq[Int], pivot1: Int, pivot2: Int): (Seq[Int], Seq[Int]) = {
    pointIdx.partition { idx =>
      val d1 = euclideanDistance(points(idx).features, points(pivot1).features)
      val d2 = euclideanDistance(points(idx).features, points(pivot2).features)
      d1 <= d2
    }
  }

  protected def makeBallTree(pointIdx: Seq[Int]): Node = {
    val mu = mean(pointIdx)
    val r = radius(pointIdx, mu)
    val ball = Ball(mu, r)
    if (pointIdx.length <= leafSize || r < epsilon) {
      //Leaf Node
      LeafNode(pointIdx, ball)
    } else {
      //split set
      val (pivot1, pivot2) = makeBallSplit(pointIdx)
      val (leftSubSet, rightSubSet) = divideSet(pointIdx, pivot1, pivot2)
      val leftChild = makeBallTree(leftSubSet)
      val rightChild = makeBallTree(rightSubSet)
      InnerNode(ball, leftChild, rightChild)
    }
  }

}

case class BallTree(override val points: IndexedSeq[VectorWithExternalId],
                    override val leafSize: Int = 50) extends Serializable with BallTreeBase {

  val root: Node = makeBallTree(pointIdx)

  private def linearSearch(query: Query, node: LeafNode): Unit = {
    val bestMatchesCandidates = node.pointIdx.map { idx =>
      BestMatch(idx, dot2(query.point,points(idx).features))
    }
    query.bestMatches ++= bestMatchesCandidates
    query.statistics.innerProductEvaluations = query.statistics.innerProductEvaluations + node.pointIdx.length
  }

  private def traverseTree(query: Query, node: Node = root): Unit = {
    if (query.bestMatches.head.value <= upperBoundMaximumInnerProduct(query, node)) {
      //This node has potential
      node match {
        case LeafNode(_, _) => linearSearch(query, node.asInstanceOf[LeafNode])
        case InnerNode(_, leftChild, rightChild) =>
          val boundLeft = upperBoundMaximumInnerProduct(query, leftChild)
          val boundRight = upperBoundMaximumInnerProduct(query, rightChild)
          if (boundLeft <= boundRight) {
            traverseTree(query, rightChild)
            traverseTree(query, leftChild)
          } else {
            traverseTree(query, leftChild)
            traverseTree(query, rightChild)
          }
        case x => throw new RuntimeException(
          s"default case in match has been visited for type${x.getClass}: " + x.toString)
      }
    } else {
      //ignoring this subtree
      query.statistics.subTreeIgnores += 1
    }
  }

  def findMultipleMaximumInnerProducts(queries: IndexedSeq[DenseVector[Double]], k: Int = 1): Seq[Seq[BestMatch]] = {
    queries.map { query: DenseVector[Double] => {
      findMaximumInnerProducts(query, k)
    }
    }
  }

  def findMaximumInnerProducts(queryPoint: DenseVector[Double], k: Int = 1): Seq[BestMatch] = {
    val query = new Query(queryPoint, k)
    query.statistics.pointsInTree = pointIdx.length
    traverseTree(query)
    val sortedBestMatches = query.bestMatches.toArray
      .filter(_.index != -1)
      .sorted(Ordering.by({ bm: BestMatch => (bm.value, bm.index) }).reverse)

    //println(query.statistics)
    //replace internal sequence id with the actual external item id
    sortedBestMatches.transform(bm => BestMatch(points(bm.index).id, bm.value))
  }

  override def toString: String = {
    s"Balltree with data size of ${points.length} (${points.take(1)})"
  }
}

case class ConditionalBallTree[T](override val points: IndexedSeq[VectorWithExternalId],
                                  labels: IndexedSeq[T],
                                  override val leafSize: Int = 50) extends Serializable with BallTreeBase {

  val root: NodeC[T] = addStats(makeBallTree(pointIdx))

  private def addStats(node: Node): NodeC[T] = {
    node match {
      case l: LeafNode =>
        LeafNodeC(
          l.pointIdx,
          l.pointIdx.map(labels).toSet,
          l.ball
        )
      case i: InnerNode =>
        val lc = addStats(i.leftChild)
        val rc = addStats(i.rightChild)
        InnerNodeC(lc.stats | rc.stats, i.ball, lc, rc)
    }
  }

  private def linearSearch(query: Query, conditioner: Set[T], node: LeafNodeC[T]): Unit = {
    val bestMatchesCandidates = node.pointIdx
      .filter(idx => conditioner(labels(idx)))
      .map(idx => BestMatch(idx, dot2(query.point, points(idx).features)))

    query.bestMatches ++= bestMatchesCandidates
    query.statistics.innerProductEvaluations = query.statistics.innerProductEvaluations + node.pointIdx.length
  }

  private def traverseTree(query: Query, conditioner: Set[T], node: NodeC[T] = root): Unit = {
    if ((node.stats & conditioner).nonEmpty &
      query.bestMatches.head.value <= upperBoundMaximumInnerProduct(query, node)) {

      //This node has potential
      node match {
        case ln: LeafNodeC[T] => linearSearch(query, conditioner, ln)
        case InnerNodeC(_, _, leftChild, rightChild) =>
          val boundLeft = upperBoundMaximumInnerProduct(query, leftChild)
          val boundRight = upperBoundMaximumInnerProduct(query, rightChild)
          if (boundLeft <= boundRight) {
            traverseTree(query, conditioner, rightChild)
            traverseTree(query, conditioner, leftChild)
          } else {
            traverseTree(query, conditioner, leftChild)
            traverseTree(query, conditioner, rightChild)
          }
        case x => throw new RuntimeException(
          s"default case in match has been visited for type${x.getClass}: " + x.toString)
      }
    } else {
      //ignoring this subtree
      query.statistics.subTreeIgnores += 1
    }
  }

  def findMultipleMaximumInnerProducts(queries: IndexedSeq[DenseVector[Double]],
                                       conditioner: Set[T],
                                       k: Int = 1): Seq[Seq[BestMatch]] = {
    queries.map { query: DenseVector[Double] => {
      findMaximumInnerProducts(query, conditioner, k)
    }
    }
  }

  def findMaximumInnerProducts(queryPoint: DenseVector[Double],
                               conditioner: Set[T],
                               k: Int = 1): Seq[BestMatch] = {
    val query = new Query(queryPoint, k)
    query.statistics.pointsInTree = pointIdx.length
    traverseTree(query, conditioner)
    val sortedBestMatches = query.bestMatches.toArray
      .filter(_.index != -1)
      .sorted(Ordering.by({ bm: BestMatch => (bm.value, bm.index) }).reverse)
    //println(query.statistics)
    //replace internal sequence id with the actual external item id
    sortedBestMatches.transform(bm => BestMatch(points(bm.index).id, bm.value))
  }

  override def toString: String = {
    s"Balltree with data size of ${points.length} (${points.take(1)})"
  }
}

