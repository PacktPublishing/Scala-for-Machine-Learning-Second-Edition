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
import org.scalaml.Predef.{DblMatrix, DblPair, DblVec}
import org.scalaml.filtering.movaverage.SimpleMovingAverage
import org.scalaml.stats.{Difference, Loss}
import org.scalaml.supervised.regression.Regression
import org.scalaml.trading.YahooFinancials
import org.scalaml.util.FormatUtils._
import org.scalaml.workflow.data.{DataSink, DataSource}
import org.scalatest.{FlatSpec, Matchers}

import scala.util.{Failure, Success, Try}

/**
 * Unit test for the Ordinary Least Square regression
 */
final class MultiLinearRegressionTest extends FlatSpec with Matchers with Resource with Logging {
  protected val name = "Ordinary least squares regression"

  final val relativePath = "supervised/regression/"

  final val SMOOTHING_PERIOD: Int = 16
  type Models = List[(Array[String], DblMatrix)]

  it should (s"$name Features extraction") in {
    show(s"$name Features extraction")

    val output = "output/regression/CNY_output.csv"
    val symbols = Array[String]("CNY", "GLD", "SPY", "TLT")

    // Selects a simple moving average to eliminate the noise from the data
    val movAvg = SimpleMovingAverage[Double](SMOOTHING_PERIOD)

    // Filter out the noise from data by applying a partial function implementing
    // as simple moving average.
    def filter(pfnMovAve: PartialFunction[DblVec, Try[DblVec]]): Try[Array[DblVec]] = Try {
      symbols.map(s => DataSource(s"${getPath(s"$relativePath$s.csv}")}", true, true, 1))
        .map(_.get.get(YahooFinancials.adjClose))
        .map(_.get.slice(20, 800))
        .map(pfnMovAve(_))
        .map(_.getOrElse(Vector.empty[Double]))
    }

    for {
      // Extract the simple moving average partial function
      pfnMovAve <- Try(movAvg |>)

      // Filter out the noise
      smoothed <- filter(pfnMovAve)

      // Generates the different models to evaluates
      models <- createModels(smoothed)

      // Compute the residual sum of square errors for each model
      rsses <- Try(getModelsRss(models, smoothed))

      // Compute the mean and total sum of square errors for each model
      (mses, tss) <- totalSquaresError(models, smoothed.head)
    } yield {
      show(s"$name ${rsses.mkString("\n")}\n${mses.mkString("\n")}\nResidual error= $tss")
    }
  }

  it should s"$name trend analysis" in {
    import Difference._, YahooFinancials._
    show(s"$name trend analysis")

    val output = "output/regression/CU_output.csv"
    val pfnSink = DataSink[Double](output) |>

    for {
      path <- getPath(s"${relativePath}CU.csv}")
      src <- DataSource(path, true, true, 1)

      prices <- src get adjClose // extract the stock price
      volatility <- src get volatility // extract the stock volatility
      volume <- src get volume // extract trading volume
      (features, expected) <- differentialData(volatility, volume, prices, diffDouble)
      regression <- MultiLinearRegression[Double](features, expected)
    } yield {
      // If the model has been created during training...
      if (regression.isModel) {
        // Save data into file
        pfnSink(expected :: volatility :: volume :: List[DblVec]())

        // Use the model created through training
        val weightsStr = regression.weights.get.view.zipWithIndex.map { case (w, n) => s"$w$n" }
        val trend = features.map(Regression.margin(_, regression.weights.get)) //w(0) + z(0)*w(1) +z(1)*w(2))

        display(expected, trend)
        show(
          s"""$name weights\n ${weightsStr.mkString(" ")}
             | \nDelta:\n${expected.mkString(",")}\nTrend: ${trend.mkString(",")}""".
            stripMargin
        )
      }
    }
  }

  private def getModelsRss(
    models: Models,
    y: Array[DblVec]
  ): List[String] = {
    models.map { case (labels, m) => s"${getRss(m.toVector, y.head, labels)}" }
  }

  private def totalSquaresError(
    models: Models,
    y: DblVec
  ): Try[(List[String], Double)] = Try {

    val errors = models.map { case (labels, m) => rssSum(m, y)._1 }
    val mses = models.zip(errors).map { case (f, e) => s"MSE for ${f._1.mkString(" ")} = $e" }

    (mses, Math.sqrt(errors.sum) / models.size)
  }

  private def getRss(xt: Vector[Array[Double]], y: DblVec, featureLabels: Array[String]): String = {
    val regression = new MultiLinearRegression[Double](xt, y)
    val modelDescriptor = regression.weights.map(_.zipWithIndex.map {
      case (w, n) =>
        val weights_str = format(w, "", SHORT)
        if (n == 0)
          s"${featureLabels(n)} = $weights_str"
        else
          s"$weights_str}.${featureLabels(n)}"
    }.mkString(" + ")).getOrElse("NA")
    s"model: $modelDescriptor\n RSS = ${regression.rss}"
  }

  def createModels(input: Array[DblVec]): Try[Models] = Try {
    val features = input.tail.map(_.toArray)

    // Retrieve the input variables by removing the first
    // time series (labeled dataset) and transpose the array
    List[(Array[String], DblMatrix)](
      (Array[String]("CNY", "SPY", "GLD", "TLT"), features.transpose),
      (Array[String]("CNY", "GLD", "TLT"), features.tail.transpose),
      (Array[String]("CNY", "SPY", "GLD"), features.take(2).transpose),
      (Array[String]("CNY", "SPY", "TLT"), features.zipWithIndex.filter(_._2 != 1)
        .map(_._1)
        .transpose),
      (Array[String]("CNY", "GLD"), features.slice(1, 2).transpose)
    )
  }

  private def rssSum(xt: DblMatrix, expected: DblVec): DblPair = {
    MultiLinearRegression[Double](xt, expected) match {
      case Success(regression) =>
        val pfnRegr = regression.|>
        (regression.rss.get, Loss.sse(expected.toArray, xt.map(pfnRegr(_).get)))
      case Failure(e) => throw new IllegalStateException(e.toString)
    }
  }

  private def display(z: DblVec, x: DblVec): Unit = {
    import org.scalaml.plots.{LinePlot, LightPlotTheme, Legend}

    val labels = Legend(
      name, "Multi-variate linear regression", s"Raw vs. filtered", "y"
    )
    val data = (z, "Delta price") :: (x, "Filtered") :: List[(DblVec, String)]()
    LinePlot.display(data, labels, new LightPlotTheme)
  }
}


// -------------------------  EOF -----------------------------------------