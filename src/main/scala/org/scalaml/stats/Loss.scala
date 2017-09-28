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
package org.scalaml.stats

import org.scalaml.Predef.Context.ToDouble
import org.scalaml.Predef._

/**
 * Generic object that encapsulates the different methodology to compute the loss or error during training
 * @author Patrick Nicolas
 * @version 0.99.2
 * @see Scala for Machine Learning Chap 3, ''Data Pre-processing'' / Time series in Scala
 */
private[scalaml] object Loss {
  import Math._
  /**
   * Compute the sum of squared errors between two arrays. It is not assumed that the arrays have identical lengths
   * @param x first array used in the sum of squared errors
   * @param z second array used in the sum of squared errors
   * @tparam T Generic type (Float, Integer, Double) for elements of arrays.
   * @return sum of squared error
   */
  def sse[@specialized(Double) T: ToDouble](x: Array[T], z: Array[T]): Double = {
    val c = implicitly[ToDouble[T]]
    val sumSqr = x.zip(z).aggregate(0.0)((s, xz) => s + sqr(c(xz._1) - c(xz._2)), _ + _)
    Math.sqrt(sumSqr)
  }

  /**
   * Compute the mean sum of squared errors between two arrays. mse = sse/number elements.
   * It is not assumed that the arrays have identical lengths
   *
   * @param x first array used in the sum of squared errors
   * @param z second array used in the sum of squared errors
   * @tparam T Generic type (Float, Integer, Double) for elements of arrays.
   * @return mean sum of squared error
   */
  def mse[@specialized(Double) T: ToDouble](x: Array[T], z: Array[T]): Double = sse(x, z) / Math.sqrt(z.length.toDouble)

  /**
   * Compute the mean sum of squared errors between two vectors of elements of type Double. mse = sse/number elements.
   * It is not assumed that the vectors have identical lengths
   *
   * @param x first array used in the sum of squared errors
   * @param z second array used in the sum of squared errorss.
   * @return mean sum of squared error
   */
  def mse(x: DblVec, z: DblVec): Double = if(x.length == 0 ) 0.0 else sse(x.toArray, z.toArray)/x.length

  /**
   * Compute the cross entropy function for the binary classification of the MLP
   * @param x first value
   * @param y second value
   * @return binary cross-entropy value
   */
  def crossEntropy(x: Double, y: Double): Double = -(x * log(y) + (1.0 - x) * log(1.0 - y))

  /**
   * Compute the cross entropy function for the binary classification of the MLP
   * @param xt first array
   * @param yt second array
   * @return cross-entropy value
   * @throws IllegalStateException if the input array have different length
   */
  @throws(classOf[IllegalArgumentException])
  def crossEntropy(xt: Array[Double], yt: Array[Double]): Double = {
    require(
      xt.length == yt.length,
      s"Loss.crossEntropy found xt length ${xt.length} and yt length ${yt.length}, should be equal"
    )
    yt.zip(xt).aggregate(0.0)({ case (s, (y, x)) => s - y * log(x) }, _ + _)
  }
}

// ----------------------------------------------   EOF ---------------------------------------------
