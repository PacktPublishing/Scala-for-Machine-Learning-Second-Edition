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
package org.scalaml.supervised.regression.linear

// Scala classes
import scala.annotation.implicitNotFound
import scala.language.implicitConversions
import scala.util.Try

import org.apache.log4j.Logger
import org.apache.commons.math3.stat.regression.SimpleRegression

import org.scalaml.Predef._
import org.scalaml.Predef.Context._
import org.scalaml.stats.TSeries
import org.scalaml.core.ITransform
import org.scalaml.util.LoggingUtils._
import TSeries._

/**
 * Class that defines the linear regression for a single variable. The model (w,r),
 * (slope, intercept) is created during instantiation of the class to reduce the life-cycle
 * of instances. The conversion of a Double back to a type T has to be defined prior
 * instantiating this class.
 * The implemantation follows the standard design of supervised learning algorithm:
 * - The classifier implements the '''ITransform''' implicit monadic data transformation
 * - The constructor triggers the training of the classifier, making the model immutable
 * - The classifier implements the '''Monitor''' interface to collect profile information for
 * debugging purpose
 *
 * {{{
 * 	regression:	w' = argmin Sum of squares {y(i)  - f(x(i)|w)}
 * 					with f(x|w) = w(0) + w(1).x
 * }}}
 * @tparam T data type of feature, bounded
 * @constructor Create a single linear regression model of type bounded to a Double as a view.
 * @see org.apache.commons.math3.stat.regression._
 * @throws IllegalArgumentException if the time series is undefined
 * @throws implicitNotFound if conversion from type to Double is not implicitly defined
 * @param xt Time series is single variable observations
 * @param expected vector of expected values or labels.
 *
 * @author Patrick Nicolas
 * @since 0.98 April 27, 2014
 * @version 0.99.2
 * @see org.scalaml.core.ITransform
 * @see org.scalaml.util.Monitor
 * @see Scala for Machine Learning Chapter 0 ''Regression and regularization'' / One variate
 * linear regression
 */
@throws(classOf[IllegalArgumentException])
@implicitNotFound("SingleLinearRegression Implicit conversion $T to Double undefined")
final private[scalaml] class SingleLinearRegression[T: ToDouble](
    xt: Vector[T],
    expected: Vector[T]
) extends ITransform[T, Double] with Monitor[Double] {

  require(
    xt.nonEmpty,
    "SingleLinearRegression. Single linear regression has undefined input"
  )

  protected val logger = Logger.getLogger("SingleLinearRegression")

  // Create the model during instantiation. The model is
  // actually create (!= None) if the regression coefficients can be computed.
  val model: Option[DblPair] = train

  /**
   * Retrieve the slope and  intercept for this single variable linear regression.
   * @return Pair (slope, intercept) of the linear regression if model has been properly trained, None otherwise
   */
  val (slope, intercept): (Option[Double], Option[Double]) = (model.map(_._1), model.map(_._2))

  @inline
  final def isModel: Boolean = model.isDefined

  /**
   * Data transformation that computes the predictive value of a time series
   * using a single variable linear regression model. The model is initialized
   * during instantiation of a class.
   * @throws MatchError if the regression model is undefined
   * @return PartialFunction of a Double value as input and the value computed using the model
   * as output
   */
  override def |> : PartialFunction[T, Try[Double]] = {
    // Compute the linear function y = slope.x + intercept
    case x: Double if model.isDefined =>
      Try(model.map { case (slope, intercept) => slope * x + intercept }.get)
  }

  private def train: Option[DblPair] = {
    // Invoke Apache commons math library for the simple regression
    val regr = new SimpleRegression(true)
    regr.addData(zipToSeries[T](xt, expected).toArray)

    // returns the slope and intercept from Apache commons math library
    Some((regr.getSlope, regr.getIntercept))
  }
}

/**
 * Companion object for the single variable linear regression. This
 * singleton is used to define the constructor for the class SingleLinearRegression.
 *
 * @author Patrick Nicolas
 * @since 0.98.1 April 27, 2014
 * @note Scala for Machine Learning Chapter 6 "Regression and regularization" / One variate
 * linear regression
 */
private[scalaml] object SingleLinearRegression {
  /**
   * Default constructor for the SingleLinearRegression class
   * @param xt Time series of (x,y) pairs of values
   * @param expected vector of expected values or labels.
   */
  @throws(classOf[IllegalArgumentException])
  @implicitNotFound("SingleLinearRegression Implicit conversion $T to Double undefined")
  def apply[T: ToDouble](
    xt: Vector[T],
    expected: Vector[T]
  ): Try[SingleLinearRegression[T]] = Try(new SingleLinearRegression[T](xt, expected))
}
// ----------------------------------  EOF ----------------------------------------