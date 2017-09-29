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
package org.scalaml.unsupervised.em

import org.scalaml.{Logging, Resource}
import org.scalaml.Predef.{DblMatrix, DblVec}
import org.scalaml.filtering.movaverage.SimpleMovingAverage
import org.scalaml.workflow.data.DataSource
import org.scalatest.{FlatSpec, Matchers}
import org.scalaml.unsupervised.env.TestEnv

import scala.util.Try


final class EMTest extends FlatSpec with Matchers with TestEnv with Logging with Resource {
  import scala.language.postfixOps

  protected[this]  val name: String = "Expectation-Maximization"
  final val rawPath = "unsupervised/em"

  it should s"$name evaluation with K=2, sampling rate=40" in {
    show(s"$name evaluation with K=2, sampling rate=40")
    execute(Array[String]("2", "40"))
  }

  it should s"$name evaluation with K=3, sampling rate=25" in {
    show(s"$name evaluation with K=3, sampling rate=25")
    execute(Array[String]("3", "25"))
  }

  it should s"$name evaluation with K=4, sampling rate=15" in {
    show(s"$name evaluation with K=4, sampling rate=15")
    execute(Array[String]("4", "15"))
  }

  it should s"$name evaluation with K=5, sampling rate=10" in {
    show(s"$name evaluation with K=5, sampling rate=10")
    execute(Array[String]("5", "10"))
  }

  private def execute(args: Array[String]): Unit = {
    require(args.length > 0, s"$name Cannot be evaluated with undefined arguments")

    val K = args(0).toInt
    val samplingRate = args(1).toInt
    val period = 8
    val smAve = SimpleMovingAverage[Double](period)
    val symbols = symbolFiles(getPath(rawPath).get)

    // extracts the observations from a set of csv files.
    assert(symbols.size > 0, s"$name The symbol files are undefined")

    // Retrieve the partial function for the simple moving average
    val pfnSmAve = smAve |>

    // Walk through the stock ticker symbols and load the historical data from each
    // ticker symbol using the data source extractor
    val obs = symbols.map(sym => {

      (for {
        path <- getPath(s"$rawPath/")
        src <- DataSource(sym, path, true, 1)

      // Extract data from the files containing historical financial data
        xs <- src |> (extractor)

        // Apply the simple moving average of the first variable
        if pfnSmAve.isDefinedAt(xs.head.toVector)
        values <- pfnSmAve(xs.head.toVector)

        // Generate the features by filtering using a sampling rate
        y <- filter(period, values, samplingRate)
      } yield y).getOrElse(Array.empty[Double])
    })
    em(K, obs)
  }

  private def filter(period: Int, values: DblVec, samplingRate: Int) = Try {
    values.view
        .indices
        .drop(period + 1).toVector
        .filter(_ % samplingRate == 0)
        .map(values(_)).toArray
  }

  private def profile(xv: Vector[Array[Double]]) {
    import org.scalaml.plots.Legend

    val legend = Legend("Density", "EM convergence K=6", "Iterations", "Density")
    val em = MultivariateEM[Double](3, xv)
    em.display("Density", legend)
  }

  private def em(K: Int, obs: DblMatrix): Int = {
    val em = MultivariateEM[Double](K, obs.toVector)
    show(s"${em.toString}")
  }
}

// -------------------------------------  EOF -------------------------------
