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

import org.apache.log4j.Logger
import org.scalaml.Predef.Context.ToDouble
import org.scalaml.Predef._
import org.scalaml.stats.Stats._
import org.scalaml.util.DisplayUtils

import scala.annotation.implicitNotFound
import scala.util.Try

/**
 * Parameterized class that computes the generic minimun and maximum of a time series. The class
 * implements:
 *
 * - Computation of minimum and maximum according to scaling factors
 *
 * - Normalization using the scaling factors
 * @tparam T type of element of the time series view bounded to a double
 * @constructor Create MinMax class for a time series of type ''Vector[T]''
 * @param values Time series of single element of type T
 * @throws IllegalArgumentException if the time series is empty
 *
 * @author Patrick Nicolas
 * @since 0.99  July 18, 2015
 * @version 0.99.2
 * @see Scala for Machine Learning Chapter 1 ''Getting Started''
 */
@implicitNotFound(msg = "MinMax conversion to Double undefined")
@throws(classOf[IllegalArgumentException])
private[scalaml] class MinMax[@specialized(Double) T: ToDouble](val values: Vector[T]) {
  import DisplayUtils._
  require(values.nonEmpty, "MinMax: Cannot initialize stats with undefined values")

  def this(values: Array[T]) = this(values.toVector)

  /**
   * Defines the scaling factors for the computation of minimum and maximum
   * @param low lower value of the range target for the normalization
   * @param high upper value of the range target for the normalization
   * @param ratio  Scaling factor between the source range and target range.
   */
  case class ScaleFactors(low: Double, high: Double, ratio: Double)

  private val logger = Logger.getLogger("MinMax")

  private[this] val zero = (Double.MaxValue, -Double.MaxValue)
  private[this] var scaleFactors: Option[ScaleFactors] = None

  /**
   * Computation of minimum and maximum values of a vector during instantiation
   */
  val (min, max): (Double, Double) = values./:(zero) { (m, x) => {
      val _x = implicitly[ToDouble[T]].apply(x)
      (if (_x < m._1) _x else m._1, if (_x > m._2) _x else m._2)
    }
  }

  @throws(classOf[IllegalStateException])
  final def normalize(low: Double = 0.0, high: Double = 1.0): DblVec =
    setScaleFactors(low, high).map(scale => {
      values.map(x => (implicitly[ToDouble[T]].apply(x) - min) * scale.ratio + scale.low)
    })
      .getOrElse(throw new IllegalStateException("MinMax.normalize normalization params undefined"))

  final def normalize(value: Double): Try[Double] = Try {
    scaleFactors.map(scale =>
      if (value <= min) scale.low
      else if (value >= max) scale.high
      else (value - min) * scale.ratio + scale.low).getOrElse(-1.0)
  }

  /**
   * Normalize the data within a range [l, h]
   * @param low lower bound for the normalization
   * @param high higher bound for the normalization
   * @return vector of values normalized over the interval [0, 1]
   * @throws IllegalArgumentException of h <= l
   */
  private def setScaleFactors(low: Double, high: Double): Option[ScaleFactors] =
    if (high < low + STATS_EPS)
      none(s"MinMax.set found high - low = $high - $low <= 0 required > ", logger)

    else {
      val ratio = (high - low) / (max - min)
      if (ratio < STATS_EPS)
        DisplayUtils.none(s"MinMax.set found ratio $ratio required > EPS ", logger)
      else {
        scaleFactors = Some(ScaleFactors(low, high, ratio))
        scaleFactors
      }
    }
}

private[scalaml] class MinMaxVector(series: Vector[Array[Double]]) {
  val minMaxVector: Vector[MinMax[Double]] = series.transpose.map(ar => {
    val x: Vector[Double] = ar
    new MinMax[Double](ar)
  })

  final def normalize(low: Double = 0.0, high: Double = 1.0): Vector[Array[Double]] =
    minMaxVector.map(_.normalize(low, high)).transpose.map(_.toArray)

  @throws(classOf[IllegalStateException])
  final def normalize(x: Array[Double]): Try[Array[Double]] = {
    val normalized = minMaxVector.zip(x).map { case (from, to) => from.normalize(to) }
    if (normalized.contains(None))
      throw new IllegalStateException("MinMax.normalize normalization params undefined")
    Try(normalized.map(_.get).toArray)
  }
}

private[scalaml] object MinMax {
  def apply[T: ToDouble](values: Vector[T]): Try[MinMax[T]] = Try(new MinMax[T](values))
}

// -------------------------  EOF -----------------------------------------