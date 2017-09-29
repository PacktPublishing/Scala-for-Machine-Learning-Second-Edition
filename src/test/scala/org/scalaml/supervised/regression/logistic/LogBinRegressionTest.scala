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

import org.scalaml.{Logging, Resource}
import org.scalaml.Predef._
import org.scalaml.stats.MinMaxVector
import org.scalaml.stats.TSeries._
import org.scalaml.trading.YahooFinancials._
import org.scalaml.util.DisplayUtils

import scala.io.Source
import scala.util.{Failure, Success, Try}
import LogBinRegression._
import DisplayUtils._
import org.scalaml.plots.{BlackPlotTheme, Legend, LinePlot, ScatterPlot}
import org.scalatest.{FlatSpec, Matchers}

/**
 * '''Purpose'''Singleton to evaluate a simple implementation of a
 * two class logistic regression for a single variable
 *
 * @author Patrick Nicolas
 * @since 0.97  November 11, 2013
 * @version 0.99.2
 * @see Scala for Machine Learning chapter 1 "Getting Started" / Let's kick the tires
 */
class LogBinRegressionTest extends FlatSpec with Matchers with Logging with Resource {

  /**
   * Name of the evaluation
   */
  val name: String = "Simple logistic Regression"
  type Fields = Array[String]

  private val NITERS = 10000
  private val EPS = 1e-6
  private val ETA = 2e-7
  private val path_training = "supervised/regression/CSCO.csv"
  private val path_test = "supervised/regression/CSCO2.csv"

  type ObsSet = (Vector[Array[Double]], Vector[Double])


  it should s"$name Simple binary logistic regression classifier" in {
    show(s"$name Simple binary logistic regression classifier")

    // Uses the for-comprehension loop to implement the computational data flow
    for {
      path <- getPath(path_training)
      // Data preparation and training
      (volatility, vol) <- load(path) // extract volatility relative to volume
      minMaxVec <- Try(new MinMaxVector(volatility)) // create a MinMax normalizer
      normVolatilityVol <- Try(minMaxVec.normalize(0.0, 1.0)) // normalize over [0, 1]
      classifier <- logRegr(normVolatilityVol, vol) // Generate the model

      // Test logistic regression
      testValues <- load(path_test) // Load the test data
      normTestValue0 <- minMaxVec.normalize(testValues._1(0)) // normalize test values
      class0 <- classifier.classify(normTestValue0)
      normTestValue1 <- minMaxVec.normalize(testValues._1(1)) // classify first data point
      class1 <- classifier.classify(normTestValue1) // classify second data point
    } yield {
      // Display the labeled data: Stock price
      val stockPrice = normalize(vol)
      displayLabel(stockPrice.get)
      Thread.sleep(20000)

      class0 should be(0, 0.466 +- 0.01)
      class1 should be(0, 0.474 +- 0.01)

      normTestValue0 should be(0.07 +- 0.005, 0)
      normTestValue0 should be(0.03 +- 0.005, 0)
      // Retrieve the model parameters (weigths)
      val result =
        s"""${toString(testValues._1(0), normTestValue0, class0)}
            |\n${toString(testValues._1(1), normTestValue1, class1)}""".stripMargin
      show(s"$name ${classifier.model.toString}\n$result")
    }
  }

  private def toString(testValue: Features, normValue: Features, res: (Int, Double)): String =
    s"$name input ${testValue.mkString(",")} normalized ${normValue.mkString(",")} class $res"

  /**
   * This method display the normalized input data in a Scatter plot
   * generates the labels and instantiate the binomial logistic regression
   */
  private def logRegr(labels: ObsSet): Try[LogBinRegression] = Try {
    val (obs, expected) = labels
    display(obs.map(x => (x(0), x(1))))

    // Extract labels and create two classes
    val normLabels = normalize(expected).getOrElse(Vector.empty[Double])
    // Generate a vector of type (DblPair, Double)
    new LogBinRegression(obs, normLabels, NITERS, ETA, EPS)
  }

  /**
   * Method to load and normalize the volume and volatility of a stock.
   */
  private def load(fileName: String): Try[ObsSet] = {
    import scala.io.Source._

    val src = fromFile(fileName)
    val data = extract(src.getLines.map(_.split(",")).drop(1))
    src.close
    data
  }

  private def extract(cols: Iterator[Fields]): Try[ObsSet] = Try {
    val features = Array[YahooFinancials](LOW, HIGH, VOLUME, OPEN, ADJ_CLOSE)
    cols.map(toArray(features)(_))
      .toVector
      .map(x => (Array[Double](1.0 - x(0) / x(1), x(2)), x(4) / x(3) - 1.0)).unzip
  }

  import scala.collection.mutable.ArrayBuffer
  private def generateLogisticRegressionData(
    numDataPoints: Int,
    tol: Double,
    dim: Int
  ): (Vector[Features], Vector[Double]) = {
    val collector = ArrayBuffer[(Array[Double], Double)]()
    (0 until numDataPoints)./:(collector)(
      (buf, idx) => {
        val rnd = new scala.util.Random(42 + idx)
        val label = if ((idx & 0x01) == 0x01) 1.0 else 0.0
        val feature = Array.fill[Double](dim) { label * tol + rnd.nextGaussian() }
        buf += ((feature, label))
      }
    ).toVector.unzip
  }

  /**
   * Method to display a time series two features: relative volatility
   * and volume
   */
  private def display(volatilityVolume: Vector[DblPair]): Unit = {
    if (isChart) {
      val info = Legend(
        "LogBinRegression",
        "LogBinRegression: CSCO 2012-13: Model features",
        "Normalized session volatility",
        "Normalized session Volume"
      )
      ScatterPlot.display(volatilityVolume, info, new BlackPlotTheme)
    }
  }

  /**
   * Method to display the label data: Stock price
   */
  private def displayLabel(price: Vector[Double]): Unit = if (isChart) {
    val info = Legend(
      "LogBinRegression",
      "CSCO 2012-13: Training label",
      "Trading sessions",
      "Normalized stock price variation"
    )
    LinePlot.display(price, info, new BlackPlotTheme)
  }

  private def displayLosses(losses: List[Double]): Unit = if (isChart) {
    val info = Legend(
      "LogBinRegression",
      "Random generator",
      "Iterations",
      "losses"
    )
    LinePlot.display(losses.toArray, info, new BlackPlotTheme)
  }

  it should s"$name evaluation with random generator" in {
    val (observations, labels) = generateLogisticRegressionData(1000, 0.01, 2)

    val regr = new LogBinRegression(observations, labels, NITERS, ETA, EPS)
    val classifiedValues = observations.zip(labels).take(20).map {
      case (observation, label) => regr.classify(observation).map(_._1).get - label
    }

    displayLosses(regr.model.losses)
    show(s"$name classification ${classifiedValues.mkString(", ")}")
  }


  it should s"$name evaluation" in {

    // Input data (2 dimension)
    val x1 = Vector[Array[Double]](
      Array[Double](0.1, 0.18),
      Array[Double](0.21, 0.17),
      Array[Double](0.45, 0.39),
      Array[Double](0.68, 0.09),
      Array[Double](0.85, 0.01),
      Array[Double](0.87, 0.73),
      Array[Double](0.59, 0.63),
      Array[Double](0.67, 0.21)
    )
    // Unormalized expected values
    val y1 = Vector[Double](1.01, 1.06, 2.49, 1.09, 0.9, 4.58, 3.81, 1.66)

    // Input data (single dimension)
    val x2 = Vector[Array[Double]](
      Array[Double](0.1),
      Array[Double](0.21),
      Array[Double](0.13),
      Array[Double](0.89),
      Array[Double](0.58),
      Array[Double](0.87),
      Array[Double](0.02),
      Array[Double](0.42),
      Array[Double](0.0)
    )
    // Unormalized expected values
    val y2 = Vector[Int](1, 2, 1, 9, 6, 9, 0, 4, 0)

    // Generation of a binomial logistic model for a two variable model
    normalize(y1) match {
      case Success(yn1) =>
        val regr = new LogBinRegression(x1, yn1, NITERS, 0.00001, 1e-8)

        val res = regr.counters(COUNT_ERR).map(_.toString).mkString("\n")
        // show(s"Counter dump $res")
        val legend = Legend(
          "Weights",
          "Binomial logistic regression convergence",
          "Recursions",
          "Error"
        )
        regr.display(List[String]("w0", "w1"), legend)

        val test0 = Array[Double](0.23, 0.19)
        val z0 = regr.classify(test0)
        z0.get._1 should be(1)
        val test1 = Array[Double](0.07, 0.71)
        val z1 = regr.classify(test1)
        z1.get._1 should be(1)
        show(s"z0 = $z0, z1 = $z1")

      case Failure(e) => error(s"$name test 2 dimension failed", e)
    }

    // Generation of a binomial logistic model for a single variable model
    normalize(y2.map(_.toDouble)) match {
      case Success(yn2) =>
        val regr = new LogBinRegression(x2, yn2, 300, 0.002, 1e-11)
        val weights = regr.model.weights

        weights.head should be(0.578 +- 0.02)
        weights.last should be(0.548 +- 0.02)
        val res = regr.counters(COUNT_ERR).map(_.toString).mkString("\n")

        show(s"$name Counter dump $res")
        val legend = Legend(
          COUNT_ERR,
          "Binomial logistic regression convergence", "Recursions", "Weights"
        )
        regr.display(COUNT_ERR, legend)

        val test0 = Array[Double](0.09)
        val z0 = regr.classify(test0)
        val test1 = Array[Double](0.91)
        val z1 = regr.classify(test1)
        show(s"z0 = $z0 z1 = $z0")

      case Failure(e) => error(s"$name test single dimension failed", e)
    }
  }
}
