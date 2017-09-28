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

import scala.util.{Try, Random}
import org.apache.log4j.Logger

import org.scalaml.Predef._
import org.scalaml.util.{LoggingUtils, MathUtils}
import LoggingUtils._, MathUtils._, LogBinRegression._

/**
 * Define the model for the binomial logistic regression: it consists of the
 * array of weights (one weight per dimension) and the intercept value
 *
 * @constructor Create a model for the binomial logistic regression
 * @param weights array of weights
 * @author Patrick Nicolas
 * @since 0.99
 * @version 0.99.2
 * @see Scala for Machine Learning Chapter 1 "Getting Started" Wrting an application
 * @see org.scalaml.supervised.regression.RegressionModel
 * @note The binomial logistic regression computes the '''intercept''' (or weight for observations
 * of zero values) independently from the optimization of the weights. Multi-nomial linear
 * and logistic regression include the intercept '''w0''' into the optimization.
 */
case class LogBinRegressionModel(weights: Features, losses: List[Double]) {
  override def toString: String = s"weights: ${weights.mkString(",")}"
}

/**
 * Logistic regression for a binary (2-class) classifier. The number of weights (model) is
 * defined by the number of variable in each observation + 1.
 * The training (extraction of the weights) is computed as part of the instantiation of the
 * class so the model is either complete or undefined so classification is never done on
 * incomplete (or poorly trained) model (computation error, maximum number of iterations
 * exceeded).
 * {{{{
 * For a vector of pair of observations and labels (x, y)
 * Likelihood (conditional probability)
 *    1/(1 + exp(-(w(0) + w(1).x(1) + w(2).x(2)))
 * Batch descent gradient formula for a learning rate eta
 *    weights <- weights - eta.(predicted - y).x
 * }}}
 *
 * @constructor Create a simple logistic regression for binary classification
 * @param expected expected values or labels used to train a model
 * @param maxIters Maximum number of iterations used during training
 * @param eta Slope used in the computation of the gradient
 * @param eps Convergence criteria used to exit the training loop.
 * @throws IllegalArgumentException if labels are undefined, or the values of maxIters, eta
 * or eps are out of range
 * @author Patrick Nicolas
 * @since 0.98 January 11, 2014
 * @version 0.99.2
 * @see Scala for Machine Learning Chapter 1 "Getting started" Let's kick the tires / Writing
 * a simple workflow
 * @see org.scalaml.supervised.regression.logistic.LogisticRegression
 * @see Design template for supervised learning algorithm Appendice / Design template for
 * classifiers
 * @note This is a simplified version of the logistic regression is presented to illustrate
 * the application. of a machine learning algorithm in chapter 1. A formal implementation of
 * the logistic regression (chap 9)  is defined in class LogisticRegression
 * @see org.scalaml.supervised.regression.logistics.LogisticRegression
 */
@throws(classOf[IllegalArgumentException])
final class LogBinRegression(
    observations: Vector[Features],
    expected: Vector[Double],
    maxIters: Int,
    val eta: Double,
    eps: Double
) extends Monitor[Double] {
  import Math._
  check(observations, expected, maxIters, eta, eps)
  protected val logger = Logger.getLogger("LogBinRegression")

  /**
   * Definition of the modelfor the binomial logistic regression
   */
  val model: LogBinRegressionModel = train

  /**
   * classification of a two dimension data (xy) using a binomial logistic regression.
   *
   * @param obs a new observation
   * @return Try (class, likelihood) for the logistic regression is the training was completed
   * , None otherwise
   */
  def classify(obs: Array[Double]): Try[(Int, Double)] = Try {
    val linear = margin(obs, model.weights) + model.weights(0)
    val prediction = sigmoid(linear)
    (if (linear >= 0.0) 1 else 0, prediction)
  }

  def classify(obs: DblPair): Try[(Int, Double)] = classify(Array[Double](obs._1, obs._2))

  // Formula M4
  private def logisticLoss(yDot: Double): Double = log(1.0 + exp(-yDot)) / observations.size
  // Formula M5
  private def derivativeLoss(y: Double, yDot: Double): Double = -y / (1.0 + exp(yDot))

  /**
   * Implements the training algorithm for the logistic regression. The model (weights)
   * is not initialized if the training loop does not converge before the maximum
   * number of iterations is reached. The training method relies on the batch gradient descent
   * which is implemented as a tail recursion
   */
  private def train: LogBinRegressionModel = {
    val labeledObs = observations.zip(expected)
    // Shuffle the data for this simplified stochastic gradient descent
    val shuffledLabeledObs = shuffle(labeledObs)

    @scala.annotation.tailrec
    def sgd(
      nIters: Int,
      weights: Weights,
      losses: List[Double]
    ): (Weights, List[Double]) =

      // if the maximum number of iterations is reached
      if (nIters >= maxIters)
        (weights, losses)
      else {
        // Traverses the (observation, label) pairs set, to compute the predicted value
        // using the logistic function (sigmoid) and compare to the labeled data.
        val (x, y) = shuffledLabeledObs(nIters % observations.size)
        val (newLoss, grad): (Double, Features) = {
          val yDot = y * margin(x, weights)
          val gradient = derivativeLoss(y, yDot)
          (logisticLoss(yDot), Array[Double](gradient) ++ x.map(_ * gradient))
        }

        // Monitor counters update
        count(COUNT_ERR, newLoss)

        if (abs(newLoss) < eps)
          (weights, losses)
        else
          // recurse to the next data point.
          sgd(
            nIters + 1,
            weights.zip(grad).map { case (w, df) => w - eta * df },
            newLoss :: losses
          )
      }

    // The weights are initialized as random values over [min labels, max labels]
    val initialWeights = Array.fill(observations.head.length + 1)(0.5)

    // Shuffle the vector of (observation, label) pairs for the next iteration
    val (weights, losses) = sgd(0, initialWeights, List[Double]())
    new LogBinRegressionModel(weights, losses)
  }

  private def addIntercept(weights: Weights): Weights = Array[Double](intercept(weights)) ++ weights.tail

  /*
	  * Computation of the intercept independently from the optimization of
	  * the weights of the logistic regression
	  */
  private def intercept(weights: Weights): Double = {
    val zeroObs = observations.filter(_.exists(_ > 0.001))
    if (zeroObs.nonEmpty)
      -zeroObs.aggregate(0.0)((s, z) => s + margin(z, weights), _ + _) / zeroObs.size
    else
      0.0
  }

  private def learningRate(nIter: Int): Double = eta / (1 + eta * nIter / observations.size)
}

/**
 * Companion object for the simple logistic regression LogBinRegression. This singleton
 * is used to define some constants and validate the class parameters.
 *
 * @author Patrick Nicolas
 * @since 0.98 January 11, 2014
 * @version 0.99.2
 * @note Add shuffling capabilities to the batch gradient descent algorithm
 */
private[scalaml] object LogBinRegression {
  final val HYPERPLANE = 0.5
  private val MAX_NUM_ITERS = 16384
  private val ETA_LIMITS = (1e-7, 1e-1)
  private val EPS_LIMITS = (1e-30, 0.01)

  final val SPAN = 6
  final val COUNT_ERR = "Error"

  type LabeledObs = Vector[(Features, Double)]
  type ObsSet = Vector[Features]
  type Weights = Array[Double]

  // dot = margin - intercept
  private val dot = (s: Double, x: (Double, Double)) => s + x._1 * x._2

  /**
   * Computes the dot product of observations and weights by adding the
   * bias element or input to the array of observations
   *
   * @param observation Array of observations (dimension of the model)
   * @param weights Array or (number of features + 1) weights
   * @return w0 + w1.x1 + .. + wn.xn
   * @throws IllegalArgumentException if obs.size != weights.size -1
   */
  @throws(classOf[IllegalArgumentException])
  final def margin(observation: Features, weights: Weights): Double = {
    require(
      observation.length + 1 == weights.length,
      s"LogBinRegression.dot found obs.length ${observation.length} +1 != weights.length ${weights.length}"
    )
    weights.tail.zip(observation.view).aggregate(weights.head)(dot, _ + _)
  }

  /**
   * Static method to shuffle the order of observations and labels between iterations
   * the gradient descent algorithm. The method is implemented as a tail recursion.
   * The shuffle is accomplish by partitioning the input data set in segments of random size
   * and reverse the order of each other segment.
   *
   * @param labeledObs input vector of pairs (multi-variable observation, label)
   * @return vector of pairs (observation, label) which order has been shuffled
   * @throws IllegalArgumentException if the labeled observations are undefined.
   */
  @throws(classOf[IllegalArgumentException])
  def shuffle(labeledObs: LabeledObs): LabeledObs = {
    require(
      labeledObs.nonEmpty,
      "LogBinRegression.shuffle Cannot proceed with undefined labeled observations"
    )

    import scala.util.Random
    def shuffle(n: Int): List[Int] =
      if (n <= 0)
        List.empty[Int]
      else
        Random.shuffle((0 until n).toList)
    shuffle(labeledObs.size).map(labeledObs(_)).toVector
  }

  private def check(
    obsSet: ObsSet,
    expected: DblVec,
    maxIters: Int,
    eta: Double,
    eps: Double
  ): Unit = {

    require(
      maxIters > 10 && maxIters < MAX_NUM_ITERS,
      s"LogBinRegression found max iterations = $maxIters required 10 < .. < $MAX_NUM_ITERS"
    )
    require(
      eta > ETA_LIMITS._1 && eta < ETA_LIMITS._2,
      s"LogBinRegression found eta = $eta requires ${ETA_LIMITS._1} < . < ${ETA_LIMITS._2}"
    )
    require(
      eps > EPS_LIMITS._1 && eps < EPS_LIMITS._2,
      s"LogBinRegression  found $eps required ${EPS_LIMITS._1} < . < ${EPS_LIMITS._2}"
    )
    require(
      obsSet.size == expected.size,
      s"LogBinRegression found ${obsSet.size} observations != ${expected.size} labels, require =="
    )
  }
}

// ----------------------  EOF ------------------------------------