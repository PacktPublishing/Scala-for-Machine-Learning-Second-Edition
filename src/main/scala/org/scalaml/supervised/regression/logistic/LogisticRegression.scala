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
package org.scalaml.supervised.regression.logistic

import scala.util.Try
import scala.language.implicitConversions
import scala.annotation.implicitNotFound

import org.apache.commons.math3.linear._
import org.apache.commons.math3.fitting.leastsquares.{LeastSquaresBuilder, LeastSquaresProblem}

import org.scalaml.core.ITransform
import org.scalaml.util.LoggingUtils._
import org.scalaml.Predef._
import org.scalaml.Predef.Context._
import org.scalaml.supervised.regression.{Regression, RegressionModel}
import org.scalaml.libraries.commonsmath.LogisticRegressionOptimizer
import org.scalaml.libraries.commonsmath.LogisticRAdapter._
import Regression._

/**
 * Logistic regression classifier. This implementation of the logistic regression does not
 * support regularization or penalty terms.
 *
 * The implementation follows the standard design of supervised learning algorithm:
 *
 * - The classifier implements the '''ITransform''' implicit monadic data transformation
 *
 * - The constructor triggers the training of the classifier, making the model immutable
 *
 * - The classifier implements the '''Monitor''' interface to collect profile information for
 * debugging purpose
 *
 * {{{
 *    The likelihood (conditional probability is computed as
 *    1/(1 + exp(-(w(0) + w(1).x(1) + w(2).x(2) + .. + w(n).x(n)))
 * }}}
 * @tparam T type of features in time series of observation.
 * @constructor Create a logistic regression classifier model.
 * @throws IllegalArgumentException if the class parameters are undefined.
 *
 * @param xt Input time series observations.
 * @param expected Labeled class data used during training of the classifier
 * @param optimizer Optimization method used to minimize the loss function during training
 * @author Patrick Nicolas
 * @since 0.98.1 May 11, 2014
 * @version 0.99.2
 * @see org.apache.commons.math3.fitting.leastsquares._
 * @see org.apache.commons.math3.optim._
 * @see org.scalaml.core.ITransform
 * @see org.scalaml.util.Monitor
 * @see Scala for Machine Learning Chapter 9 ''Regression and regularization'' / Logistic regression
 */
@throws(classOf[IllegalArgumentException])
@implicitNotFound(msg = "LogisticRegression Implicit conversion $T to Double undefined")
final private[scalaml] class LogisticRegression[@specialized(Double) T: ToDouble](
    xt: Vector[Array[T]],
    expected: Vector[Int],
    optimizer: LogisticRegressionOptimizer
) extends ITransform[Array[T], Int] with Regression with Monitor[Double] {

  import LogisticRegression._

  check(xt, expected)

  /**
   * Binary predictor using the Binomial logistic regression and implemented
   * as a data transformation (PipeOperator). The predictor relies on a margin
   * error to associated to the outcome 0 or 1.
   * @throws MatchError if the model is undefined or has an incorrect size or the input feature
   * is undefined
   * @return PartialFunction of feature of type Array[T] as input and the predicted class as output
   */
  override def |> : PartialFunction[Array[T], Try[Int]] = {
    case x: Array[T] if isModel && model.get.size - 1 == x.length => Try {
      if (margin(x, model.get) > HYPERPLANE) 1 else 0
    }
  }

  /**
   * Training method for the logistic regression
   */
  override protected def train: RegressionModel = {
    val weights0 = Array.fill(xt.head.length + 1)(INITIAL_WEIGHT)

    /*
		 * Anonymous Class that defines the computation of the value of
		 * the function and its derivative (Jacobian matrix) for all the data points.
		 */
    val lrJacobian = new RegressionJacobian(xt, weights0)

    /**
     * Create an instance of the convergence criteria (or exit strategy)
     * using the Apache Commons Math ConvergenceChecker
     */
    val exitCheck = new RegressionConvergence(optimizer)

    /**
     * The Apache Commons Math lib. require to set up the optimization using
     * A builder. The Builder in turn, generates a Least Square problem 'lsp'
     */
    def createBuilder: LeastSquaresProblem =
      (new LeastSquaresBuilder).model(lrJacobian)
        .weight(MatrixUtils.createRealDiagonalMatrix(Array.fill(xt.size)(1.0)))
        .target(expected.toArray)
        .checkerPair(exitCheck)
        .maxEvaluations(optimizer.maxEvals)
        .start(weights0)
        .maxIterations(optimizer.maxIters)
        .build

    val optimum = optimizer.optimize(createBuilder)
    RegressionModel(optimum.getPoint.toArray, optimum.getRMS)
  }
}

/**
 * Companion object for the logistic regression. The singleton is used
 * for conversion between Apache Common Math Pair Scala tuple and vice versa.
 * The singleton is also used to define the constructors
 * @author Patrick Nicolas
 * @since 0.98. 1May 11, 2014
 * @version 0.99.2
 * @see Scala for Machine Learning Chapter 7 Regression and regularization/Logistic regression
 */
private[scalaml] object LogisticRegression {

  /**
   * Class class discriminant for the logit function
   */
  final val INITIAL_WEIGHT = 0.5

  /**
   * Bias or margin used for one of the class in the logistic regression
   */
  final val MARGIN = 0.01

  /**
   * Adjusted class discriminant value for the linear exponent fo the logit
   */
  final val HYPERPLANE = -Math.log(1.0 / (MARGIN + INITIAL_WEIGHT) - 1)

  /**
   * Default constructor for the logistic regression
   * @param xt Input time series observations.
   * @param expected Labeled class data used during training of the classifier
   * @param optimizer Optimization method used to minimize the loss function during training
   * @return Logistic regression instance
   */
  def apply[T: ToDouble](
    xt: Vector[Array[T]],
    expected: Vector[Int],
    optimizer: LogisticRegressionOptimizer
  ): Try[LogisticRegression[T]] = Try(new LogisticRegression[T](xt, expected, optimizer))

  private def check[T](xt: Vector[Array[T]], expected: Vector[Int]): Unit = {
    require(
      xt.nonEmpty,
      "Cannot compute the logistic regression of undefined time series"
    )
    require(
      xt.size == expected.size,
      s"Size of input data ${xt.size} is different from size of labels ${expected.size}"
    )
  }
}

// --------------------------------------  EOF -------------------------------------------------------