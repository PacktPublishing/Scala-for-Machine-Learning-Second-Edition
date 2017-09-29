/**
 * Copyright (c) 2013-2017  Patrick Nicolas - Scala for Machine Learning - All rights reserved
 *
 * Licensed under the Apache License, Version 2.0 (the "License") you may not use this file
 * except in compliance with the License. You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *
 * The source code in this file is provided by the author for the sole purpose of illustrating the
 * concepts and algorithms presented in "Scala for Machine Learning 2nd edition".
 * ISBN: 978-1-783355-874-2 Packt Publishing.
 *
 * Version 0.99.2
 */
package org.scalaml.unsupervised.functionapprox

/**
 * Bin for histogram defines as a mean feature value, average target value _y and the
 * number of data point for this bin
 * @author Patrick Nicolas
 * @version 0.99.2
 * @param y average value for the bin
 */
final private[scalaml] class Bin(var y: Double = 0.0) extends Serializable {
  var count: Int = 0

  /**
   * Adds a new data point a the mid point of this bin or interval
   * The average ''_y'' is automatically recomputed
   * @param newY new value added to this bin
   */
  def +=(newY: Double): this.type = {
    val newCount = count + 1
    y = if (count == 0) newY else (y * count + newY) / newCount
    count = newCount
    this
  }

  /**
   * Merge this bin with another bin. The method returns the reference to this
   * instance to avoid unnecessary cloning
   * @param next Bin to be merged with this bin
   * @return reference to this bin
   * @see codingchallenge.Histogram.--
   */
  def +(next: Bin): this.type = {
    val newCount = count + next.count
    if (newCount > 0) y = (y * count + next.y * next.count) / newCount
    this
  }

  // final def equals(other: Bin): Boolean = Math.abs(x - other.x) < 1e-3 && Math.abs(y - other.y) < 1e-3

  /**
   * Merge an array of bins with this bin. The method returns the reference to this
   * instance to avoid unnecessary cloning
   * @param next Array of bins to be merged with this bin
   * @return reference to this bin
   */
  def +(next: Array[Bin]): this.type = {
    val newCount = next.aggregate(count)((s, nxt) => s + nxt.count, _ + _)
    if (newCount > 0) {
      y = next./:(y * count)((s, nxt) => s + nxt.y * nxt.count) / newCount
      count = newCount
    }
    this
  }

  override def toString: String = s"${y}/t$count"
}

// -----------------------------------  EOF -----------------------------------------