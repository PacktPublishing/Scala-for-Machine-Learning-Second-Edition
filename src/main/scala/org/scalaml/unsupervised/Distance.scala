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
package org.scalaml.unsupervised

import org.scalaml.Predef.Context._

/**
 * Singleton which defines the different distances used by unsupervised machine learning
 * techniques.
 * @author Patrick Nicolas
 * @since 0.98 February 16, 2014
 * @version 0.99.2
 * @see Scala for Machine Learning Chapter 4 "Unsupervised Learning" Measuring similarity
 */
private[scalaml] object Distance {
  private final val sqr = (x: Double) => x * x

  /**
   * Function that compute the Manhattan distance between two
   * array (vectors ) of values.
   * @tparam T type of elements of the 1st vector used in the computation of the distance
   * @tparam U type of elements of the 2nd vector used in the computation of the distance
   * @param x first array/vector/data point
   * @param y second array/vector/data point
   * @throws IllegalArgumentException if the input array are undefined or have different size
   * @return distance between two data points
   */
  @throws(classOf[IllegalArgumentException])
  def manhattan[@specialized(Double) T: ToDouble, @specialized(Double) U: ToDouble](
    x: Array[T],
    y: Array[U]
  ): Double = {
    require(
      x.length == y.length,
      s"Distance.manhattan Vectors have different size ${x.length} and ${y.length}"
    )
    val cu: ToDouble[T] = implicitly[ToDouble[T]]
    val cv: ToDouble[U] = implicitly[ToDouble[U]]
    (x, y).zipped.map { case (u, v) => Math.abs(cu.apply(u) - cv.apply(v)) }.sum
  }

  /**
   * Function that compute the Euclidean distance between two
   * array (vectors ) of values.
   * @tparam T type of elements of the 1st vector used in the computation of the distance
   * @tparam U type of elements of the 2nd vector used in the computation of the distance
   * @param x first array/vector/data point
   * @param y second array/vector/data point
   * @throws IllegalArgumentException if the input array are undefined or have different size
   * @return distance between two data points
   */
  @throws(classOf[IllegalArgumentException])
  def euclidean[@specialized(Double) T: ToDouble, @specialized(Double) U: ToDouble](
    x: Array[T],
    y: Array[U]
  ): Double = {
    require(
      x.length == y.length,
      s"Distance.euclidean Vectors have different size ${x.length} and ${y.length}"
    )
    val cu: ToDouble[T] = implicitly[ToDouble[T]]
    val cv: ToDouble[U] = implicitly[ToDouble[U]]
    Math.sqrt((x, y).zipped.map { case (u, v) => sqr(cu.apply(u) - cv.apply(v)) }.sum)
  }

  /**
   * Function that compute the normalized inner product or cosine distance between two
   * vectors
   * @tparam T type of elements of the 1st vector used in the computation of the distance
   * @tparam U type of elements of the 2nd vector used in the computation of the distance
   * @param x first array/vector/data point
   * @param y second array/vector/data point
   * @throws IllegalArgumentException if the input array are undefined or have different size
   * @return distance between two data points
   */
  @throws(classOf[IllegalArgumentException])
  def cosine[@specialized(Double) T: ToDouble, @specialized(Double) U: ToDouble](
    x: Array[T],
    y: Array[U]
  ): Double = {
    require(
      x.length == y.length,
      s"Distance.cosine Vectors have different size ${x.length} and ${y.length}"
    )

    val cu: ToDouble[T] = implicitly[ToDouble[T]]
    val cv: ToDouble[U] = implicitly[ToDouble[U]]
    val norms = (x, y).zipped.map {
      case (u, v) =>
        val wu = cu.apply(u)
        val wv = cv.apply(v)
        Array[Double](wu * wv, wu * wu, wv * wv)
    }./:(Array.fill(3)(0.0))((s, t) => s ++ t)

    norms(0) / Math.sqrt(norms(1) * norms(2))
  }
}

// -------------------------------  EOF ----------------------------------------------------