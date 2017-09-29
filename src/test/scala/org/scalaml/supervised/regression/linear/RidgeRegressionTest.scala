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

import org.scalaml.{Logging, Resource}
import org.scalaml.workflow.data.DataSource
import org.scalatest.{FlatSpec, Matchers}
import org.scalaml.supervised.regression.Regression
import org.scalaml.stats.{Difference, TSeries}
import org.scalaml.trading.YahooFinancials._
import Difference._
import org.scalaml.Predef.Context.ToDouble
import org.scalaml.Predef.DblVec
import org.scalaml.util.FormatUtils._

import scala.util.Try

/**
 * Created by patrick.nicolas on 3/6/17.
 */
final class RidgeRegressionTest extends FlatSpec with Matchers with Resource with Logging {
  import Regression._
  protected val name = "Ridge Regression "
  val relativePath = "supervised/regression/CU.csv"

  private val dataInput = "output/supervised/regression/CU_input.csv"
  final val LAMBDA: Double = 0.5

  it should s"$name L2 regularization" in {
    show(s"$name L2 regularization")

    for {
      path <- getPath(relativePath)
      // Select the data source
      src <- DataSource(path, true, true, 1)
      // Load historical stock price
      price <- src.get(adjClose)

      // Load then compute the historical stock relative volatility within a trading session
      volatility <- src.get(volatility)

      // Load then compute the historical stock relative trading volume
      volume <- src.get(volume)

      // Extract the features value and labels through differentiation
      (features, expected) <- differentialData(volatility, volume, price, diffDouble)

      // Generate a regression model using L2 penalty
      regression <- RidgeRegression[Double](features, expected, LAMBDA)
    } yield {

      // Use the model created through training
      if (regression.isModel) {
        val weightsStr = regression.weights.get.view.zipWithIndex
          .map { case (w, n) => s"${n}-${format(w, ": ", SHORT)}" }
        val trend = features.map(margin(_, regression.weights.get))

        show(s"""$name Weights\n ${weightsStr.mkString(" ")}
                						| \nDelta:\n${expected.mkString(",")}\nTrend: ${trend.mkString(",")}
                						| ${format(regression.rss.get, "rss =", MEDIUM)}""".stripMargin)

        // Create two predictors to be evaluated
        val y1 = predict(0.2, expected, volatility, volume)
        val y2 = predict(5.0, expected, volatility, volume)

        // Generate the prediction for different values of the regularization factor..
        val output = (2 until 10 by 2).map(n => {
          val lambda = n * 0.1
          val y = predict(lambda, expected, volatility, volume)
          s"\nLambda  $lambda\n${format(y, "", SHORT)}"
        })
        show(output.mkString("."))
      } else
        error(s"$name evaluation failed")
    }
  }

  private def predict(
    lambda: Double,
    expected: DblVec,
    volatility: DblVec,
    volume: DblVec
  ): DblVec = {
    import TSeries._


    val observations = zipToSeries(volatility, volume, 1)
    val regression = new RidgeRegression[Double](observations, expected, lambda)
    val fnRegr = regression |>

    observations.map(x => if (fnRegr.isDefinedAt(x)) fnRegr(x).get else Double.NaN)
  }

}


// -------------------------------  EOF -----------------------------------