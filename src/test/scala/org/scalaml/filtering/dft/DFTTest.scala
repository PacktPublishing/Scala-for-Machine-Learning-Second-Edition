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
package org.scalaml.filtering.dft

import org.scalaml.{Logging, Resource}
import org.scalaml.Predef.DblVec
import org.scalaml.trading.YahooFinancials._
import org.scalaml.util.FormatUtils.{SHORT, format}
import org.scalaml.workflow.data.{DataSink, DataSource}
import org.scalatest.{FlatSpec, Matchers}

import scala.util.Try

/**
  *
  */
final class DFTTest extends FlatSpec with Matchers with Logging with Resource {
  protected[this] val name: String = "Discrete Fourier transform"

  final private val RESOURCE_PATH = "filtering"
  final private val OUTPUT1 = "output/filtering/simulated.csv"
  final private val OUTPUT2 = "output/filtering/smoothed.csv"
  final private val OUTPUT3 = "output/filtering/filt_"

  final private val FREQ_SIZE = 1025
  final private val DISPLAY_SIZE = 128

  final private val F = Array[Double](0.005, 0.05, 0.2)
  final private val A = Array[Double](2.0, 1.0, 0.5)

  private def harmonic(x: Double, n: Int): Double = A(n) * Math.cos(Math.PI * F(n) * x)
  private val h = (x: Double) =>
    Range(0, A.size).aggregate(0.0)((s, i) => s + harmonic(x, i), _ + _)


  it should s"$name with synthetic data" in {
    import scala.language.postfixOps
    show("$name with synthetic data")
    val INV_FREQ = 1.0 / FREQ_SIZE

    // Extract the frequencies spectrum using the discrete Fourier transform
    // and write into output file
    val pfnDFT = DFT[Double] |>

    // A simple pipeline for generating harmonics, store the output data into a sink file
    // build the frequency spectrum and save it into file.
    (for {
      values <- Try { Vector.tabulate(FREQ_SIZE)(n => h(n * INV_FREQ)) }
      output1 <- DataSink[Double](OUTPUT1) write values

      if pfnDFT.isDefinedAt(values)
      frequencies <- pfnDFT(values)
      output2 <- DataSink[Double](OUTPUT2) write frequencies
    } yield {

      // Display the 2nd to 128th frequencies
      val displayed = frequencies.take(DISPLAY_SIZE)
      display(displayed.drop(1))

      val results = format(displayed, "x/1025", SHORT)
      show(s"$name $DISPLAY_SIZE frequencies: $results")
    }).getOrElse(-1)
  }




  it should s"$name DFT filter for BAC price history" in {
    import org.scalaml.filtering.dft.DTransform._
    val CUTOFF = 0.005
    val CUTOFF2 = 0.01
    val symbol = "BAC"

   // val path = getPath(s"$RESOURCE_PATH/$symbol.csv")
   // val src = DataSource(s"$RESOURCE_PATH/$symbol.csv", false, true, 1)

    // Generate the partial function for the low pass filter
    val pfnDFTfilter = DFTFilter[Double](CUTOFF)(sinc) |>
    val pfnDFTfilter2 = DFTFilter[Double](CUTOFF2)(sinc) |>

    // Extract the daily adjusted stock closing price
    // then applies the DFT filter
    (for {
      path <- getPath(s"$RESOURCE_PATH/$symbol.csv")
      src <- DataSource(path, false, true, 1)
      price <- src.get(adjClose)

      // Applies the first DFT filter with  CUTOFF = 0.005
      if pfnDFTfilter.isDefinedAt(price)
      filtered <- pfnDFTfilter(price)

      // Applies the second DFT filter with  CUTOFF = 0.01
      if pfnDFTfilter2.isDefinedAt(price)
      filtered2 <- pfnDFTfilter2(price)
    } yield {
      // Store filtered data in output file
      val sink2 = DataSink[Double](s"output/filtering/$symbol.csv")
      sink2 |> filtered :: List[DblVec]()

      // Display the low pass filter with CUTOFF
      display(price, filtered, CUTOFF)

      // Display the low pass filter with CUTOFF2
      display(price, filtered2, CUTOFF2)
      val result = format(filtered.take(DISPLAY_SIZE), "DTF filtered", SHORT)
      show(s"First $DISPLAY_SIZE frequencies: $result")
    }).get
  }


  private def display(data: DblVec): Unit = {
    import org.scalaml.plots.{Legend, LightPlotTheme, LinePlot}

    val title = s"Discrete Fourier Frequencies 1 - $DISPLAY_SIZE"
    val labels = Legend(name, title, "Frequencies", "Amplitude")
    LinePlot.display(data, labels, new LightPlotTheme)
  }

  /*
		 * Display two time series of frequencies with a predefined cutOff value fc
		 * using the LinePlot class
		 */
  private def display(x1: DblVec, x2: DblVec, fc: Double): Unit = {
    import org.scalaml.plots.{Legend, LightPlotTheme, LinePlot}

    var labels = Legend(
      name,
      "Discrete Fourier low pass filter",
      s"Low pass frequency $fc",
      "Stock price"
    )
    val _x2 = x2.take(x1.size - 96)
    val data = (x1, "Stock price") :: (_x2, "DFT") :: List[(DblVec, String)]()
    LinePlot.display(data, labels, new LightPlotTheme)

    labels = Legend(
      name,
      "Discrete Fourier low pass filter - Noise",
      s"Low pass frequency $fc",
      "delta price"
    )
    val data2 = x1.zip(_x2).map { case (x, y) => x - y }
    LinePlot.display(data2, labels, new LightPlotTheme)
  }
}
