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
package org.scalaml.filtering.movingaverage

import org.scalaml.{Logging, Resource}
import org.scalaml.Predef.DblVec
import org.scalaml.filtering.movaverage.{ExpMovingAverage, SimpleMovingAverage, WeightedMovingAverage}
import org.scalaml.trading.YahooFinancials.adjClose
import org.scalaml.util.{Assertable, FormatUtils}
import org.scalaml.workflow.data.{DataSink, DataSource}
import org.scalatest.{FlatSpec, Matchers}

import scala.util.Try


final class MovingAverageTest extends FlatSpec with Matchers with Logging with Assertable with Resource {
  import FormatUtils._
  protected[this] val name: String = "Moving average smoothing"

  private val START_DISPLAY = 32
  private val WINDOW_DISPLAY = 64
  private val RESOURCE_PATH = "filtering/"
  private val OUTPUT_PATH = "output/filtering/mvaverage"

  it should s"$name evaluation of stock prices" in  {
    show(s"$name evaluation of stock prices")

    val symbol = "BAC"
    val p = 60
    val p_2 = p >> 1
    val w = Array.tabulate(p)(n => if (n == p_2) 1.0 else 1.0 / (Math.abs(n - p_2) + 1))
    val sum = w.sum
    val weights: Array[Double] = w.map { _ / sum }

    // Extract the partial functions associated to the simple, weighted and
    // exponential moving averages
    val pfnSMvAve = SimpleMovingAverage[Double](p) |>
    val pfnWMvAve = WeightedMovingAverage[Double](weights) |>
    val pfnEMvAve = ExpMovingAverage[Double](p) |>

    val results: Try[Int] = for {
      path <- getPath(s"$RESOURCE_PATH$symbol.csv")
      dataSrc <- DataSource(path, false)
      price <- dataSrc.get(adjClose)
      // Executes the simple moving average
      if pfnSMvAve.isDefinedAt(price)
      sMvOut <- pfnSMvAve(price)

      // Executes the exponential moving average
      if pfnEMvAve.isDefinedAt(price)
      eMvOut <- pfnEMvAve(price)

      // Executes the weighted moving average
      if pfnWMvAve.isDefinedAt(price)
      wMvOut <- pfnWMvAve(price)
    } yield {
      val dataSink = DataSink[Double](s"$OUTPUT_PATH$p.csv")

      // Collect the results to be saved into a data sink (CSV file)
      val results = List[DblVec](price, sMvOut, eMvOut, wMvOut)
      dataSink |> results

      show(s"Results for [$START_DISPLAY, $WINDOW_DISPLAY] values")

      results.map(window(_)).map(display(_))
      display(price, sMvOut, "Simple Moving Average")
      display(price, eMvOut, "Exponential Moving Average")
      display(price, wMvOut, "Weighted Moving Average")
    }
    results.get
  }



  final val PERIOD = 3
  final val WEIGHTS = Array[Double](0.2, 0.6, 0.2)

  final val input = Vector[Double](
    1.0, 2.0, 1.5, 2.0, 2.8, 3.2, 3.9, 4.6, 5.2, 4.0, 3.7, 3.1, 2.5, 2.2, 1.6, 1.2, 0.4
  )

  final val SIMPLE_MOVAVG_LABEL = Vector[Double](
    0.0, 0.0, 1.50, 1.83, 2.10, 2.66, 3.30, 3.90, 4.57, 4.60, 4.30, 3.6, 3.10, 2.60, 2.11, 1.67, 1.07
  )

  final val EXP_MOVAVG_LABEL = Vector[Double](
    1.00, 1.50, 1.50, 1.75, 2.27, 2.74, 3.32, 3.96, 4.58, 4.29, 3.99, 3.55, 3.02, 2.61, 2.11, 1.65, 1.03
  )

  final val WEIGHTED_MOVAVG_LABEL = Vector[Double](
    0.00, 0.00, 1.70, 1.70, 2.06, 2.72, 3.26, 3.90, 4.58, 4.84, 4.18, 3.64, 3.10, 2.56, 2.14, 1.64, 1.12, 0.48
  )


  it should s"$name comparison of moving average techniques" in {
    show("$name comparison of moving average techniques")

    val assertMsg = "Moving average tests"
    import scala.language.postfixOps

      // Extract the partial smoothing functions
    val pfnSMvAve = SimpleMovingAverage[Double](PERIOD) |>
    val pfnWMvAve = WeightedMovingAverage[Double](WEIGHTS) |>
    val pfnEMvAve = ExpMovingAverage[Double](PERIOD) |>

      // Generates the smoothed data
    (for {
      sMvOut <- pfnSMvAve(input)
      eMvOut <- pfnEMvAve(input)
      wMvOut <- pfnWMvAve(input)
    } yield {
      compareInt(sMvOut.size, input.size)
      compareVector(sMvOut, SIMPLE_MOVAVG_LABEL, 1e-1)
      show(s"$name simple ${sMvOut.mkString(",")}")

      compareInt(eMvOut.size, input.size)
      compareVector(eMvOut, EXP_MOVAVG_LABEL, 1e-1)
      show(s"$name exponential ${eMvOut.mkString(",")}")

      compareInt(wMvOut.size, input.size)
      compareVector(wMvOut, WEIGHTED_MOVAVG_LABEL, 1e-1)
      show(s"$name weighted ${wMvOut.mkString(",")}")
    }).getOrElse(-1)
  }


  private def window(series: DblVec): DblVec =
    series.drop(START_DISPLAY).take(WINDOW_DISPLAY)

  private def display(values: DblVec): Unit = show(format(values, "X", SHORT))

  private def display(price: DblVec, smoothed: DblVec, label: String): Int = {
    import org.scalaml.plots.{LinePlot, LightPlotTheme, Legend}

    val labels = Legend(name, label, "Trading sessions", "Stock price")

    val dataPoints = List[DblVec](price, smoothed).map(_.toVector).zip(labels.toList)
    LinePlot.display(dataPoints.toList, labels, new LightPlotTheme)
    0
  }

}

// -------------------------------------------  EOF -------------------------------------
