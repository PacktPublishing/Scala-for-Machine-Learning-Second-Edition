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
package org.scalaml.filtering.kalman

import org.scalaml.{Logging, Predef, Resource}
import org.scalaml.Predef.DblVec
import org.scalaml.stats.TSeries.zipWithShift
import org.scalaml.trading.YahooFinancials
import org.scalaml.trading.YahooFinancials.adjClose
import org.scalaml.util.Assertable
import org.scalaml.util.FormatUtils.{LONG, format}
import org.scalaml.workflow.data.{DataSink, DataSource}
import org.scalatest.{FlatSpec, Matchers}

import scala.util.{Failure, Random, Success}


final class DKalmanTest extends FlatSpec with Matchers with Logging with Assertable with Resource {
  protected[this] val name: String = "Kalman filter"

  private val OUTPUT_FILE = "output/filtering/dkalman"
  private val RESOURCE_DIR = "filtering/"
  private val NUM_VALUES = 128

  // Noise has to be declared implicitly
  implicit val qrNoise = new QRNoise((0.7, 0.3), (m: Double) => m * Random.nextGaussian)
  // Contract extractor
  private val extractor = YahooFinancials.adjClose :: List[Array[String] => Double]()

  it should s"$name evaluation" in {
    import Predef._

    show(s"$name evaluation")

    // H and P0 are the only components that are independent from
    // input data and smoothing factor. The control matrix B is not defined
    // as there is no external control on the time series.
    val H: DblMatrix = ((0.9, 0.0), (0.0, 0.1))
    val P0: DblMatrix = ((0.4, 0.3), (0.5, 0.4))

    /**
      * Inner function that updates the parameters/matrices for a two-step lag
      * Kalman filter.
      */
    def twoStepLagSmoother(xSeries: DblVec, alpha: Double): Int = {
      require(alpha > 0.0 && alpha < 1.0, s"Smoothing factor $alpha is out of range")

      // Generates the A state transition matrix from the times series updating equation
      val A: DblMatrix = ((alpha, 1.0 - alpha), (0.9, 0.1))

      // The control matrix is null
      val B: DblMatrix = ((0.0, 0.0), (0.0, 0.0))

      // Generate the state as a time series of pair [x(t+1), x(t)]
      val xt: Vector[(Double, Double)] = zipWithShift(xSeries, 1)
      val DISPLAY_LENGTH = 20
      show(s"$name First $DISPLAY_LENGTH data points x[t+1] - x[t]\n${xt.take(DISPLAY_LENGTH).mkString("\n")}")

      val pfnKalman = DKalman(A, H, P0) |>

      // Applied the Kalman smoothing for [x(t+1), x(t)]
      pfnKalman(xt) match {
        case Success(filtered) => {
          // Dump results in output file along the original time series
          val output = s"${OUTPUT_FILE}_${alpha.toString}.csv"
          val results = filtered.map(_._1)

          // Illustration of usage of the data sink
          DataSink[Double](output) |> results :: xSeries :: List[DblVec]()

          // For convenience, on the first NUM_VALUES are plotted
          val displayedResults = results.take(NUM_VALUES)
          display(xSeries, results, alpha)

          // Formatted rsults
          val result = format(
            displayedResults.toVector,
            s"2-step lag smoother first $NUM_VALUES values", LONG
          )
          show(s"$name results $result\nCompleted")
        }
        case Failure(e) => error(s"$name failed with ${e.toString}")
      }
    }

    val symbol = "BAC"
    for {
      path <- getPath(s"$RESOURCE_DIR$symbol.csv")
      source <- DataSource(path, false)
    } yield {
      // Evaluate two different step lag smoothers.
      source.get(adjClose).map(zt => {
        twoStepLagSmoother(zt, 0.4)
        twoStepLagSmoother(zt, 0.7)
      })
      1
    }
  }


  /*
   * Ubiquitous method to plot two single variable time series using
   * org.scalaml.plot.LinePlot class
   */
  private def display(z: DblVec, x: DblVec, alpha: Double): Unit = {
    import org.scalaml.plots.{LinePlot, LightPlotTheme, Legend}

    val labels = Legend(
      name, s"Kalman filter alpha = $alpha", s"Kalman with alpha $alpha", "y"
    )
    val data = (z, "price") :: (x, "Filtered") :: List[(DblVec, String)]()
    LinePlot.display(data, labels, new LightPlotTheme)
  }

}


// --------------------------------  EOF ----------------------------------------------------
