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
package org.scalaml.util

import org.scalaml.stats.TSeries._

/**
 * Singleton that define the comparison between data sets with different types
 *
 * @author Patrick Nicolas
 * @since 0.99 June 13, 2015
 * @version 0.99.2
 */
trait Assertable {

  /**
   * Method that compare multi-dimensional data set of type VSeries
   * @param predicted Predicted or computed values to be evaluated
   * @param expected Expected value or labels to be compared with
   * @param eps error criteria
   * @throws IllegalArgumentException if the input time series have different size
   * @return true if each element of the two time series are similar within the error range
   * @note The ''IllegalArgumentException'' is thrown by '''zipToVSeries''' XTSeries method
   */
  @throws(classOf[IllegalArgumentException])
  protected def compareVectorArray(
    predicted: Vector[Array[Double]],
    expected: Vector[Array[Double]],
    eps: Double
  ): Boolean = {

    val xCompare = (x: Double, y: Double) => Math.abs(x - y)
    val fCompare = (x: Array[Double], y: Array[Double]) =>
      zipToArray(x, y)(xCompare).sum

    zipToVSeries(predicted, expected)(fCompare).forall(_ < eps)
  }

  /**
   * Method that compare one-dimensional data set of type VSeries
   * @param predicted Predicted or computed values to be evaluated
   * @param expected Expected value or labels to be compared with
   * @param eps error criteria
   * @throws IllegalArgumentException if the input time series have different size
   * @return true if each element of the two time series are similar within the error range
   * @note The ''IllegalArgumentException'' is thrown by '''zipToVector''' XTSeries method
   */
  @throws(classOf[IllegalArgumentException])
  protected def compareVector(
    predicted: Vector[Double],
    expected: Vector[Double],
    eps: Double
  ): Boolean = {

    val xCompare = (x: Double, y: Double) => Math.abs(x - y)
    zipToVector(predicted, expected)(xCompare).forall(_ < eps)
  }

  /**
   * Method that compare one-dimensional data set of type VSeries
   * @param predicted Predicted or computed values to be evaluated
   * @param expected Expected value or labels to be compared with
   * @param eps error criteria
   * @throws IllegalArgumentException if the input time series have different size
   * @return true if each element of the two time series are similar within the error range
   * @note The ''IllegalArgumentException'' is thrown by '''zipToVector''' XTSeries method
   */
  @throws(classOf[IllegalArgumentException])
  protected def compareArray(
    predicted: Array[Double],
    expected: Array[Double],
    eps: Double
  ): Boolean = compareVector(predicted.toVector, expected.toVector, eps)

  /**
   * Method that compares two values of type Double
   * @param predicted Predicted or computed value
   * @param expected Expected value
   * @param eps error criteria
   * @return true if the value are similar within the error range
   */
  protected def compareDouble(predicted: Double, expected: Double, eps: Double): Boolean = {
    Math.abs(predicted - expected) < eps
  }

  /**
   * Method that compares two values of type Int
   * @param predicted Predicted or computed value
   * @param expected Expected value
   * @return true if the values are equals
   */
  protected def compareInt(predicted: Int, expected: Int): Boolean = predicted == expected

  def assertVector[T](predicted: Vector[T], expected: Vector[T]): Boolean =
    predicted.zip(expected.view).forall { case (p, e) => p == e }

  def assertList[T](predicted: List[T], expected: List[T]): Unit =
    assertVector(predicted.toVector, expected.toVector)

  def assertT[T](predicted: T, expected: T): Boolean = predicted == expected
}

// ---------------------------- EOF ----------------------