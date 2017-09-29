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
import org.scalatest.{FlatSpec, Matchers}

import scala.language.postfixOps
import org.scalaml.trading.YahooFinancials._
import org.scalaml.stats.{Difference, Loss}
import org.scalaml.util.FormatUtils._
import org.scalaml.workflow.data.DataSource
import Difference._
import org.apache.commons.math3.fitting.leastsquares.LevenbergMarquardtOptimizer
import org.scalaml.libraries.commonsmath.LogisticRegressionOptimizer

import scala.util.{Failure, Success, Try}

final class LogisticRegressionTest extends FlatSpec with Matchers with Resource with Logging {
  protected[this] val name = "Logistic Regression classifier";

  private val maxIters = 250
  private val maxEvals = 4500
  private val eps = 1e-7
  final val relativePath: String = "supervised/regression/CU.csv"

  it should s"$name evaluation" in {
    show(s"$name evaluation")

    // Select the non-linear optimizer for minimizing the loss function
    val optimizer = new LevenbergMarquardtOptimizer()
    for {
      path <- getPath(relativePath)
      // Instantiate a new data source
      src <- DataSource(path, true, true, 1)
      // Load historical stock price
      price <- src.get(adjClose)

      // Load historical stock relative volatility
      volatility <- src.get(volatility)

      // Load historical stock relative volume
      volume <- src.get(volume)

      // Extract the features value and labels through differentiation
      (features, expected) <- differentialData(volatility, volume, price, diffInt)

      // Set the configuration for the mimization of the loss function
      lsOpt <- LogisticRegressionOptimizer(maxIters, maxEvals, eps, optimizer)

      // Create a logistic regression model through training using features as input
      // data and expected as labels
      regr <- LogisticRegression[Double](features, expected, lsOpt)

      // Extract the partial function logistic predictor
      pfnRegr <- Try(regr |>)
    } yield {
      // Generates prediction  values..
      show(s"$name ${toString(regr)}")
      val accuracy = features.map(pfnRegr(_)).zip(expected).map {
        case (tryP, e) => tryP.map(p => if (p == e) 1 else 0).getOrElse(-1)
      }.sum / expected.size.toDouble

      show(s"$name Accuracy: $accuracy")
    }
  }

  private def toString(regression: LogisticRegression[Double]): String =
    s"""Regression model ---:\n 
       		| ${format(regression.rss.getOrElse(-1.0), "Rss", SHORT)}\nWeights:
       		| ${
      regression.weights
        .map(format(_, "", MEDIUM))
        .mkString(" ")
    }""".stripMargin
}
