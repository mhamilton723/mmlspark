package com.microsoft.ml.spark.nn

import java.io.Serializable
import breeze.linalg.DenseVector

final case class InnerNode(override val ball: Ball,
                           leftChild: Node,
                           rightChild: Node) extends Node {
  override def toString: String = {
    s"InnerNode with ${ball.toString}}."
  }
}

final case class LeafNode(pointIdx: Seq[Int],
                          override val ball: Ball) extends Node {
  override def toString: String = {
    s"LeafNode with ${ball.toString}} \n " +
      s"and data size of ${pointIdx.length} (example point: ${pointIdx.take(1)}})"
  }
}

trait Node extends Serializable {
  def ball: Ball
}

final case class InnerNodeC(override val stats: Map[String, Int],
                            override val ball: Ball,
                            leftChild: NodeC,
                            rightChild: NodeC) extends NodeC {
  override def toString: String = {
    s"InnerNode with ${ball.toString} and ${stats.toString}."
  }
}

final case class LeafNodeC(pointIdx: Seq[Int],
                           override val stats: Map[String, Int],
                           override val ball: Ball) extends NodeC {
  override def toString: String = {
    s"LeafNode with ${ball.toString} and ${stats.toString} \n " +
      s"and data size of ${pointIdx.length} (example point: ${pointIdx.take(1)}})"
  }
}

trait NodeC extends Node {
  def stats: Map[String, Int]
}

final case class Ball(mu: DenseVector[Double], radius: Double) extends Serializable

final case class BestMatch(index: Int, value: Double) extends Serializable

final case class VectorWithExternalId(id: Int, features: DenseVector[Double]) extends Serializable {
  override def toString: String = {
    s"IdWithFeatures(${id}},${features}})"
  }
}
