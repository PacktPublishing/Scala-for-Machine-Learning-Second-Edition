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
import org.scalaml.Predef.DblVec
import org.scalaml.stats.Loss
import org.scalaml.trading.YahooFinancials
import org.scalaml.util.FormatUtils._
import org.scalaml.workflow.data.DataSource
import org.scalatest.{FlatSpec, Matchers}

import scala.util.Try

/**
 * '''Purpose:''' Singleton to evaluatcdorge the single variate linear regression.
 *
 * The test consists of creating a linear regression y = slope.x + intercept for a stock price
 *
 * @author Patrick Nicolas
 * @version 0.99.2
 * @see Scala for Machine Learning Chapter 9: ''Regression and regularization'' / One-variate
 * linear regression
 */
final class SingleLinearRegressionTest extends FlatSpec with Matchers with Resource with Logging {
  protected val name = "Single linear regression"

  it should s"$name Linear regression on stock price" in {
    import YahooFinancials._
    show(s"$name Linear regression on stock price")

    for {
      path <- getPath("supervised/regression/CU.csv")
      // Price is the expected values
      src <- DataSource(path, false, true, 1)

      price <- src.get(adjClose)

      // Generates the trading days for X-axis and create
      // a single linear regression model
      days <- Try(Vector.tabulate(price.size)(_.toDouble))
      linearRegr <- SingleLinearRegression[Double](days, price)
    } yield {
      // Make sure we have a valid regression model
      linearRegr.model.foreach { m =>
        val slope = m._1
        val intercept = m._2
        // Display the raw and regressed prices on a line plot.
        val values = predict(days, price, slope, intercept).unzip
        val entries = List[DblVec](values._1, values._2)

        display(entries.zip(List[String]("raw price", "Linear regression")))

        // Dump model parameters and tabulate data into standard out.
        val results =
          s"""Regression ${format(slope, "y= ", SHORT)}.x + ${format(intercept, " ", SHORT)}
               | Mean square error: ${mse(days, price, slope, intercept)}""".stripMargin

        show(s"$name $results\nPredicted   Expected\n${tabulate(days, price, slope, intercept)}")
      }
    }
  }

  private def display(values: List[(DblVec, String)]): Unit = {
    import org.scalaml.plots.{LinePlot, LightPlotTheme, Legend}

    val legend = Legend(
      name, "Single linear regression stock price", "Trading sessions", "Prices"
    )
    LinePlot.display(values, legend, new LightPlotTheme)
  }

  private def predict(
    xt: DblVec,
    expected: DblVec,
    slope: Double,
    intercept: Double
  ): Vector[(Double, Double)] =
    xt.view.zip(expected.view).map { case (x, y) => (slope * x + intercept, y) }.toVector

  private def tabulate(
    xt: DblVec,
    expected: DblVec,
    slope: Double,
    intercept: Double
  ): String = predict(xt, expected, slope, intercept).map {
    case (p, e) => s"${slope * p + intercept}, $e"
  }.mkString("\n")

  private def mse(
    xt: DblVec,
    expected: DblVec,
    slope: Double,
    intercept: Double
  ): Double = Loss.mse(xt.map(slope * _ + intercept), expected)
}


// -----------------------------------  EOF -----------------------------------------
