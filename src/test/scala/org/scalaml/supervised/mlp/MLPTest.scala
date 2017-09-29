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
package org.scalaml.supervised.mlp

import org.scalaml.{Logging, Resource}
import org.scalaml.Predef.DblVec
import org.scalaml.workflow.data.DataSource
import org.scalatest.{FlatSpec, Matchers}
import org.scalaml.trading.GoogleFinancials
import GoogleFinancials._
import org.scalaml.plots.{Legend, LightPlotTheme, LinePlot}

import scala.collection.mutable.ArrayBuffer
import scala.util.Random


final class MLPTest extends FlatSpec with Matchers with Resource with Logging {
  protected[this] val name: String = "Multi-layer perceptron"

  it should s"$name bimodal classification with one hidden layer" in {
    show(s"$name bimodal classification with one hidden layer")
    evaluate(Array[Int](4))
  }

  it should s"$name bimodal classification with two hidden layer" in {
    show(s"$name bimodal classification with two hidden layer")
    evaluate(Array[Int](3, 3))
  }

  it should s"$name Binary classification two hiddenlayers impact of alpha and eta factors" in {
    import scala.language.postfixOps
    show(s"Binary classification two hiddenlayers impact of alpha and eta factors")

    val ALPHA = 0.3
    val ETA = 0.03
    val HIDDEN = Array[Int](7, 3)
    val NUM_EPOCHS = 2000
    val TEST_SIZE: Int = 100
    val EPS = 1e-7

    def testEta(
      eta: Double,
      xt: Vector[Array[Double]],
      yt: Vector[Array[Double]]): ArrayBuffer[(Double, String)] = {

      implicit val mlpObjective = new MLP.MLPBinClassifier
      val config = MLPConfig(ALPHA, eta, NUM_EPOCHS, EPS)
      MLP[Double](config, HIDDEN, xt, yt).counters("err").map((_, s"eta=$eta"))
    }

    def testAlpha(
      alpha: Double,
      xt: Vector[Array[Double]],
      yt: Vector[Array[Double]]): ArrayBuffer[(Double, String)] = {
      implicit val mlpObjective = new MLP.MLPBinClassifier
      val config = MLPConfig(alpha, ETA, NUM_EPOCHS, EPS)
      MLP[Double](config, HIDDEN, xt, yt).counters("err").map((_, s"alpha=$alpha"))
    }

    def f1(x: Double): Array[Double] =
      Array[Double](0.1 + 0.5 * Random.nextDouble, 0.6 * Random.nextDouble)

    def f2(x: Double): Array[Double] =
      Array[Double](0.6 + 0.4 * Random.nextDouble, 1.0 - 0.5 * Random.nextDouble)


    // Generate the synthetic time series of features
    val HALF_TEST_SIZE = TEST_SIZE >> 1
    val xt = Vector.tabulate(TEST_SIZE)(n =>
      if (n < HALF_TEST_SIZE) f1(n) else f2(n - HALF_TEST_SIZE))

    show(s"$name input data ${xt.map(_.mkString(",")).mkString("\n")}")

    // Generate the synthetic expected values (labels)
    val yt = Vector.tabulate(TEST_SIZE)(n =>
      if (n < HALF_TEST_SIZE) Array[Double](0.0) else Array[Double](1.0)
    )

    val etaValues = List[Double](0.005, 0.02, 0.03, 0.1)
    val data = etaValues.flatMap(testEta(_, xt, yt)).map{ case (x, s) => (Vector(x), s) }
    val values = data.map(_._1(0))
    show(s"Output values\n${values.take(50).mkString("\n")}")

    val alphaValues = List[Double](0.0, 0.2, 0.4, 0.6)
    val data2 = alphaValues.flatMap(testAlpha(_, xt, yt)).map{ case (x, s) => (Vector(x), s)}
    val values2 = data2.map(_._1(0))
    show(s"Output values\n${values2.take(50).mkString("\n")}")
  }


  private def evaluate(hiddenLayers: Array[Int]): Unit = {
    val dataPath = "supervised/mlp"
    val ALPHA = 0.8
    val ETA = 0.05
    val NUM_EPOCHS = 2500
    val EPS = 1e-6
    val THRESHOLD = 0.25

    val symbols = Array[String](
      "FXE", "FXA", "SPY", "GLD", "FXB", "FXF", "FXC", "FXY", "CYB"
    )

    val STUDIES = List[Array[String]](
      Array[String]("FXY", "FXC", "GLD", "FXA"),
      Array[String]("FXE", "FXF", "FXB", "CYB"),
      Array[String]("FXE", "FXC", "GLD", "FXA", "FXY", "FXB"),
      Array[String]("FXC", "FXY", "FXA"),
      Array[String]("CYB", "GLD", "FXY")
       /*
      ,
      symbols
      */
    )

    implicit val mode = new MLP.MLPBinClassifier

    val desc =
      s"""$name classifier without SoftMax conversion
         			| ${hiddenLayers.mkString(" ")} hidden layers""".stripMargin
    show(desc)

    val prices = symbols.map(s => {
      val path = getPath(s"$dataPath/$s.csv")
      DataSource( s"${path.getOrElse(".")}", true, true, 1)
    }).map( _.flatMap(_.get(close))).filter(_.isSuccess).map(_.get)

    show(s"$name Data input size: ${prices(0).length}")
    test(hiddenLayers, prices)


    def test(hiddenLayers: Array[Int], prices: Array[DblVec]): Int  =  {
      show(s"$name ${hiddenLayers.size} layers:(${hiddenLayers.mkString(" ")})")

      val startTime = System.currentTimeMillis
      val config = MLPConfig(ALPHA, ETA, NUM_EPOCHS, EPS)

      STUDIES.foreach(eval(hiddenLayers, prices, config, _))
      show(s"$name Duration ${System.currentTimeMillis - startTime} msecs.")
    }

    def eval(
      hiddenLayers: Array[Int],
      obs: Array[DblVec],
      config: MLPConfig,
      etfsSet: Array[String]): Int  =
      fit(hiddenLayers, etfsSet, obs, config).map(acc => show(s"Accuracy: $acc"))
      .getOrElse(error(s"$name could not compute the accuracy"))

    def fit(
      hiddenLayers: Array[Int],
      symbols: Array[String],
      prices: Array[DblVec],
      config: MLPConfig): Option[Double] = {
      val obs = symbols.flatMap( index.get(_)).map(prices(_).toArray)

      val xv = obs.tail.transpose
      val expected = Array[Array[Double]](obs.head).transpose

      val classifier = MLP[Double](config, hiddenLayers, xv, expected)

      classifier.fit(THRESHOLD)
    }


    def index: Map[String, Int] =  {
      import scala.collection.mutable.HashMap
      symbols.zipWithIndex./:(HashMap[String, Int]())((mp, si)  => mp += ((si._1, si._2))).toMap
    }

    def toString(symbols: Array[String]): String =
      s"${symbols.tail.mkString(" ")} s => ${symbols(0)}"
  }
}


// --------------------------------   EOF ---------------------------------------------------------