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

import scala.annotation.implicitNotFound
import scala.collection.immutable.VectorBuilder
import scala.util.Try
import scala.reflect.ClassTag
import scala.language.implicitConversions
import org.apache.log4j.Logger

import org.scalaml.Predef._
import org.scalaml.Predef.Context._

/**
 * Companion object for time series, that defined the essential functions on time series such
 * as statistics, transpose, zScore, normalize.... This singleton implements basic type
 * conversion (implicit)
 * @author Patrick Nicolas
 * @since 0.98 January, 22, 2014
 * @version 0.99.2
 * @see Scala for Machine Learning Chapter 3 "Data pre-processing" Time series
 */
private[scalaml] object TSeries {
  final val EPS = 1 - 20

  private val logger = Logger.getLogger("XTSeries")

  /**
   * Generate a time series of tuple of type (T,T) by zipping the
   * - last ''size - n'' observations of this time series with the
   * - first ''size - n'' observations
   * @tparam T type of element (or data point or observation) of the time series
   * @param xv Single variable vector to be zipped
   * @param n Number of observations used in the shift
   * @throws IllegalArgumentException if the shift parameter n is out of range
   * @return time series of pairs from shifted Vectors.
   */
  @throws(classOf[IllegalArgumentException])
  def zipWithShift[T](xv: Vector[T], n: Int): Vector[(T, T)] = {
    require(
      n > 0 && n < xv.size,
      s"XTSeries.zipWithShift found shift n= $n required n < ${xv.size}"
    )
    xv.drop(n).zip(xv.view.dropRight(n))
  }

  /**
   * Generate a time series of tuple of type (T,T) by zipping the
   * last ''size - n'' observations of this time series with the
   * first ''size - n'' observations
   * @tparam T type of element (or data point or observation) of the time series
   * @param xv Single variable array to be zipped
   * @param n Number of observations used in the shift
   * @throws IllegalArgumentException if the shift parameter n is out of range
   * @return time series of pairs from shifted arrays
   */
  @throws(classOf[IllegalArgumentException])
  def zipWithShift[T](xv: Array[T], n: Int): Array[(T, T)] = {
    require(
      n > 0 && n < xv.length,
      s"XTSeries.zipWithShift found shift n= $n required n < ${xv.length}"
    )

    xv.drop(n).zip(xv.view.dropRight(n))
  }

  /**
   * Generate a time series of tuple of type (T,T) by zipping the time series
   * with itself
   * @tparam T type of element (or data point or observation) of the time series
   * @param xv Single variable vector to be zipped
   * @return time series of pairs from shifted Vectors.
   */
  def zipWithShift1[T](xv: Vector[T]): Vector[(T, T)] = xv.zip(xv.view.tail)

  /**
   * Generate a time series of tuple of type (T,T) by zipping the time series
   * with itself
   * @tparam T type of element (or data point or observation) of the time series
   * @param xv Single variable vector to be zipped
   * @return time series of pairs from shifted Vectors.
   */
  def zipWithShift1[T](xv: Array[T]): Array[(T, T)] = xv.zip(xv.view.tail)

  /**
   * Splits this time series into two distinct time series at a given index '''n'''
   * @tparam T type of element (or data point or observation) of the time series
   * @param xv Multi-variable vector to be splitted
   * @param n index in the time series used in the split
   * @return 2-tuple or pair of times series ''(ts1, ts2)'' ts1 containing n first elements
   * in the original time series, ts2 containing the remaining elements
   * @throws IllegalArgumentException if parameter n is out of bounds
   */
  @throws(classOf[IllegalArgumentException])
  def splitAt[T](xv: Vector[T], n: Int): (Vector[T], Vector[T]) = {
    require(n > 0 && n < xv.size, s"XTSeries.splitAt found index $n required index < ${xv.size}")

    val splitArr = xv.splitAt(n)
    (splitArr._1, splitArr._2)
  }

  /**
   * Retrieve the dimension of the time series that is the number of variable in
   * each observations or data points
   * @tparam T type of element (or data point or observation) of the time series
   * @param xt time series of arrays
   * @return number of features in the observations of the time series
   */
  def dimension[T](xt: Vector[Array[T]]): Int = xt.head.length

  /**
   * Implements the normalization of a parameterized time series
   * @tparam T type of element (or data point or observation) of the time series
   * @param xt single dimension parameterized time series
   * @return normalized time series as double elements if max > min, None otherwise
   */
  def normalize[T: ToDouble](
    xt: Vector[T],
    low: Double,
    high: Double
  )(implicit ordering: Ordering[T]): Try[DblVec] = Try(Stats[T](xt).normalize(low, high))

  def normalize[T: ToDouble](xt: Vector[T])(implicit ordering: Ordering[T]): Try[DblVec] =
    normalize(xt, 0.0, 1.0)

  /**
   * Implements the normalization of a parameterized multi-dimension time series within [0, 1]
   * @tparam T type of element (or data point or observation) of the time series
   * @param xt multi-dimension parameterized time series
   * @throws IllegalArgumentException if the time series is undefined
   * @return normalized time series as double elements if max > min, None otherwise
   */
  @throws(classOf[IllegalArgumentException])
  @implicitNotFound(msg = "XTSeries.normalize conversion from $T to Double undefined")
  def normalizeArray[T: ToDouble](
    xt: Vector[Array[T]]
  )(implicit order: Ordering[T], m: Manifest[T]): Try[DblSeries] = {
    require(
      xt.nonEmpty,
      "TSeries.normalize Cannot normalize an undefined time series of elements"
    )
    require(
      dimension(xt) > 0,
      "TSeries.normalize Incorrect function to normalize a single dimension time series"
    )

    var k = 0
    val res = new Array[Array[T]](xt.size)
    val dim = dimension(xt)

    val min = Array.fill(dim)(Double.MaxValue)
    val max = Array.fill(dim)(-Double.MaxValue)

    val convert: ToDouble[T] = implicitly[ToDouble[T]]
    // computes min and max
    while (k < xt.size) {
      var j = 0
      while (j < dim) {
        val temp: Double = convert(xt(k)(j))
        if (temp < min(j))
          min(j) = temp
        else if (temp > max(j))
          max(j) = temp
        j += 1
      }
      k += 1
    }

    val data = new VectorBuilder[Array[Double]]
    k = 0

    Try {
      while (k < xt.size) {
        var j = 0
        val arr = new Array[Double](dim)
        while (j < dim) {
          arr(j) = (convert(xt(k)(j)) - min(j)) / (max(j) - min(j))
          j += 1
        }
        data += arr
        k += 1
      }
      data.result()
    }
  }

  def normalize(xt: DblVec, low: Double, high: Double): Try[DblVec] =
    normalize[Double](xt, low, high)
  def normalizeArray(xt: Vector[Array[Double]], low: Double, high: Double): Try[DblSeries] =
    normalizeArray(xt, low, high)

  def zScore[T: ToDouble](xt: Vector[T]): Try[DblVec] = Try(Stats[T](xt).zScore)

  /**
   * transform time series of parameterized array into a array of double Vector
   * by applying the '''Z score''' transform to each element of the time series.
   * @tparam T type of element (or data point or observation) of the time series
   * @param xt multi-dimensional parameterized time series
   * @throws IllegalArgumentException if the time series is undefined
   * @return Time series of double array if the function succeeds, None otherwise
   */
  @throws(classOf[IllegalArgumentException])
  def zScores[T: ToDouble](xt: Vector[Array[T]]): Try[Vector[Array[Double]]] = {
    require(xt.nonEmpty, "zScores Cannot zScore an undefined time series")
    import scala.collection.immutable.VectorBuilder
    val convert: ToDouble[T] = implicitly[ToDouble[T]]

    val stats = statistics(xt)
    var k = 0
    val dimension = xt.head.length
    val data = new VectorBuilder[Features]

    Try {
      while (k < xt.size) {
        var j = 0
        val arr = Array.fill(dimension)(0.0)
        while (j < dimension) {
          arr(j) = (convert(xt(k)(j)) - stats(j).mean) / stats(j).stdDev
          j += 1
        }
        data += arr
        k += 1
      }
      data.result()
    }
  }

  def unit[T: ToDouble](xt: Vector[Array[T]]): Try[Vector[Array[Double]]] = {
    val c: ToDouble[T] = implicitly[ToDouble[T]]
    Try(xt.map(_.map(c(_))))
  }

  def zipToVSeries[T](x: VSeries[T], y: VSeries[T])(f: (Array[T], Array[T]) => Double): Vector[Double] = {
    require(
      x.size == y.size,
      s"XTSeries.zipSeries found x.size = ${x.size} != y.size  ${y.size}"
    )
    x.zip(y.view).map { case (_x, _y) => f(_x, _y) }
  }

  def zipToVector[T](x: Vector[T], y: Vector[T])(f: (T, T) => Double): Vector[Double] = {
    require(
      x.size == y.size,
      s"XTSeries.zipSeries found x.size = ${x.size} != y.size  ${y.size}"
    )
    x.zip(y.view).map { case (_x, _y) => f(_x, _y) }
  }

  def zipToArray[T](x: Array[T], y: Array[T])(f: (T, T) => Double): Array[Double] = {
    require(
      x.length == y.length,
      s"VSeries.zipSeries found x.length = ${x.length} != y.length  ${y.length}"
    )
    x.zip(y.view).map { case (_x, _y) => f(_x, _y) }
  }

  def zipToSeries[T: ToDouble](x: Vector[T], y: Vector[T]): DblSeries = {
    require(
      x.size == y.size,
      s"VSeries.zipSeries found x.size = ${x.size} != y.size  ${y.size}"
    )
    val convert: ToDouble[T] = implicitly[ToDouble[T]]
    x.zip(y.view).map { case (_x, _y) => Array[Double](convert(_x), convert(_y)) }
  }

  def zipToSeries[T: ClassTag](x: Vector[T], y: Vector[T], nSteps: Int)(implicit f: T => Double): DblSeries = {
    require(nSteps > 0, s"VSeries.zipSeries found nSteps = $nSteps, required > 0")
    require(
      x.size == y.size,
      s"VSeries.zipSeries found x.size = ${x.size} != y.size  ${y.size}"
    )

    x.zip(y.view).map { case (_x, _y) => Array[Double](_x, _y) }.dropRight(nSteps)
  }

  def margin[T: ToDouble](xt: Array[T], zt: Array[Double]): Double = {
    val convert: ToDouble[T] = implicitly[ToDouble[T]]
    xt.zip(zt).map { case (_x, z) => convert(_x) * z }.sum
  }


  def innerDouble(xt: Array[Double], zt: Array[Double]): Double =
    xt.zip(zt)./:(0.0) { case (s, (x, z)) => s + x * z }

  def transform[T: ClassTag](xt: VSeries[T]): Try[VSeries[T]] = Try(xt.transpose.map(_.toArray))

  /**
   * Compute the basic aggregate statistics for a time series
   * @tparam T type of element (or data point or observation) of the time series
   * @param xt time series for which the statistics are computed
   * @return Statistics instance
   */
  def statistics[T: ToDouble](xt: Vector[T]): Stats[T] = Stats[T](xt)

  /**
   * Compute the basic statistics for each dimension of a time series
   * @tparam T type of element (or data point or observation) of the time series
   * @param xt time series for which the statistics are computed
   * @return Array of statistics for each dimension
   */
  @throws(classOf[IllegalArgumentException])
  def statistics[T: ToDouble](xt: Vector[Array[T]]): Vector[Stats[T]] = {
    require(xt.nonEmpty || dimension(xt) > 0, "XTSeries.statistics input time series undefined")
    xt.transpose.map(Stats[T](_))
  }

  def statisticsDouble(xt: Vector[Array[Double]]): Vector[Stats[Double]] = statistics[Double](xt)
}

// ---------------------------------  EOF --------------------------------------------------------