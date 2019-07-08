// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.ml.spark.vw

import com.microsoft.ml.spark.core.contracts.Wrappable
import org.apache.spark.ml.param.{BooleanParam, IntParam}

/** Controls hashing parameters such us number of bits (numbits) and how to handle collisions.
  */
trait HasNumBits extends Wrappable {
  val numbits = new IntParam(this, "numbits", "Number of bits used to mask")
  setDefault(numbits -> 30)

  def getNumBits: Int = $(numbits)
  def setNumBits(value: Int): this.type = {
    if (value < 1 || value > 30)
      throw new IllegalArgumentException("Number of bits must be between 1 and 30 bits")
    set(numbits, value)
  }

  /**
    * @return the bitmask used to constrain the hash feature indices.
    */
  protected def getMask: Int = ((1 << getNumBits) - 1)

  val sumCollisions = new BooleanParam(this, "sumCollisions", "Sums collisions if true, otherwise removes them")
  setDefault(sumCollisions -> true)

  def getSumCollisions: Boolean = $(sumCollisions)
  def setSumCollisions(value: Boolean): this.type = set(sumCollisions, value)
}
