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
package org.scalaml.supervised.svm

import org.scalaml.{Logging, Resource}
import org.scalaml.Predef.{DblPair, DblVec}
import org.scalaml.supervised.svm.formulation.{CSVCFormulation, OneSVCFormulation, SVRFormulation}
import org.scalaml.supervised.svm.kernel._
import org.scalaml.workflow.data.DataSource
import org.scalatest.{FlatSpec, Matchers}
import org.scalaml.stats.Transpose._
import org.scalaml.trading.Fundamentals._
import org.scalaml.stats.TSeries._
import org.scalaml.supervised.regression.linear.SingleLinearRegression
import org.scalaml.util.FormatUtils._

import scala.collection.mutable.ArrayBuffer
import scala.util.{Random, Try}


final class SVMTest extends FlatSpec with Matchers with Logging with Resource {
  protected[this] val name: String = "Support Vector Machine"

  type Features = Vector[Array[Double]]

  /**
    * Evaluate SVM as a binary classifier
    */
  it should s"$name evaluation binary classifier" in {
    show(s"Evaluation binary classifier")

    val relativePath = "supervised/svm/dividends2.csv"
    val C = 1.0
    val GAMMA = 0.5
    val EPS = 1e-2
    val NFOLDS = 2

    val extractor = relPriceChange ::
      debtToEquity ::
      dividendCoverage ::
      cashPerShareToPrice ::
      epsTrend ::
      shortInterest ::
      dividendTrend ::
      List[Array[String] =>Double]()

   // val pfnSrc = DataSource(path, true, false, 1).map( _.|>)
    val config = SVMConfig(new CSVCFormulation(C), new RbfKernel(GAMMA), SVMExecution(EPS, NFOLDS))

    /**
      * Generate the matrix of observations by feature from the data
      * extracted from the data source (CSV file)
      */
    def getObservations(input: Features): Try[Features] = Try {
      val vec = input.dropRight(1).map(_.toArray)
      transpose(vec).toVector
    }

    (for {
      path <- getPath(relativePath)
      pfnSrc <- DataSource(path, true, false, 1)
      pfn <- pfnSrc.|> (extractor)
      obs <- getObservations(pfn)
    } yield {
      val svc = SVM[Double](config, obs, pfn.last.toVector)
      show(s"$name ${svc.toString}\naccuracy ${svc.accuracy.getOrElse(-1.0)}")
    })
    .getOrElse( error(s"$name classifier failed"))
  }

  /**
    * Compare multiple kernel functions for various profiles for input data
    */
  it should s"$name evaluation kernel functions" in {
    show(s"Evaluation kernel functions")

    // Generic parameters for the support vector machines
    val EPS = 0.001
    val C = 1.0
    val GAMMA = 0.8
    val N = 100
    val COEF0 = 0.5
    val DEGREE = 2

    compareKernel(0.6, 0.3)
    compareKernel(0.7, 0.3)
    compareKernel(0.8, 0.3)
    compareKernel(1.3, 0.3)

    def genData(variance: Double, mean: Double): Features = {
      val rGen = new scala.util.Random(System.currentTimeMillis)

      Vector.tabulate(N)(_ => {
        rGen.setSeed(rGen.nextLong)
        Array[Double](rGen.nextDouble, rGen.nextDouble).map(variance*_ - mean)
      })
    }

    // Computes the accuracy of the classifier on a synthetic test sets to
    // evaluate and compare multiple kernel functions.

    def compareKernel(a: Double, b: Double): Int = {
      require(a > 0.0 && a < 2.0, s"$name Cannot compare Kernel with inadequate features a = $a")
      require(b > -0.5 && b < 0.5, s"$name Cannot compare Kernel with inadequate features b = $b")

      // We make sure that the synthetic training and validation set are extracted from
      // the same dataset, sharing the same data distribution, genData
      val trainingSet = genData(a, b) ++ genData(a, 1-b)
      val testSet = genData(a, b) ++ genData(a, 1-b)
      val setHalfSize = trainingSet.size>>1

      val legend = s"\n$name Skewed Random values: a=$a b= $b"
      val validationSet = trainingSet.drop(setHalfSize)
      display(legend,
        trainingSet.take(setHalfSize).map(z => (z(0), z(1))),
        validationSet.map(z => (z(0), z(1)))
      )

      val labels: DblVec = Vector.fill(N)(0.0) ++ Vector.fill(N)(1.0)

      val accuracies = accuracy(trainingSet, testSet, labels, new RbfKernel(GAMMA)) ::
        accuracy(trainingSet, testSet, labels, new SigmoidKernel(GAMMA)) ::
        accuracy(trainingSet, testSet, labels, LinearKernel) ::
        accuracy(trainingSet, testSet, labels, new PolynomialKernel(GAMMA, COEF0, DEGREE)) ::
        List[Double]()

      val kernelType = List[String](
        s"RBF($GAMMA)", s"Sigmoid($GAMMA)", "Linear", s"Polynomial($GAMMA, $COEF0, $DEGREE)"
      )
      val result = accuracies.zip(kernelType).map{case(ac, ks) => s"$ks\t\t$ac}" }.mkString("\n")
      show(s"\n$name Comparison of kernel functions on synthetic data $legend\n$result")
    }


    /**
      * Compute the accuracy of the SVM classifier given a kernel function.
      * The accuracy is computed as the percentage of true positives
      */
    def accuracy(
      xt: Features,
      test: Features,
      labels: DblVec,
      kF: SVMKernel): Double = {

      import scala.language.postfixOps

      // Configuration and instantiation (training) of a support vector machine classifier
      val config = SVMConfig(new CSVCFormulation(C), kF)
      val svc = SVM[Double](config, xt, labels)

      // Retrieve the predictive partial function
      val pfnSvc = svc |>

      // Count the true positives
      test.zip(labels).count{ case(x, y) =>
        if(pfnSvc.isDefinedAt(x)) pfnSvc(x).get == y else false}.toDouble/test.size
    }

    def display(label: String, xy1: Vector[DblPair], xy2: Vector[DblPair]): Boolean = {
      import org.scalaml.plots.{ScatterPlot, BlackPlotTheme, Legend}
      val labels = Legend(name, s"SVM Kernel evaluation", label, "Y")
      ScatterPlot.display(xy1, xy2, labels, new BlackPlotTheme)
    }
  }

  /**
    * Evaluate the impact of the value of the margin on the accuracy of the SVM classifier
    */
  it should s"$name evaluation of impact of margin" in {
    show("Evaluation of impact of margin")

    val GAMMA = 0.8
    val CACHE_SIZE = 1<<8
    val NFOLDS = 2
    val EPS = 1e-5
    val N = 100
    var status: Int = 0

    generate.map( values => {
      val result = (0.1 until 5.0 by 0.1)
                   .flatMap( evalMargin(values._1, values._2, _) ).mkString("\n")

      show(s"\nMargin with\nC\tMargin\n$result")
    }).getOrElse(-1)

    def generate: Option[(Features, DblVec)] = {
      val z: Features  = Vector.tabulate(N)(i =>
        Array[Double](i, i*(1.0 + 0.2*Random.nextDouble))
      ) ++
        Vector.tabulate(N)(i => Array[Double](i, i*Random.nextDouble))
      normalizeArray(z).map( (_, Vector.fill(N)(1.0) ++ Vector.fill(N)(0.0))).toOption
    }

    /**
      * Evaluate the impact of the margin factor C over the accuracy of the SVM classifier
      */
    def evalMargin(
      features: Features,
      expected: DblVec,
      c: Double): Option[String] = {

      // Set up the configuration
      val execEnv = SVMExecution(CACHE_SIZE, EPS, NFOLDS)
      val config = SVMConfig(new CSVCFormulation(c), new RbfKernel(GAMMA), execEnv)

      // Instantiate the SVM classifier and train the model
      val svc = SVM[Double](config, features, expected)

      // Extract and stringize the margin for a given C penalty value.
      svc.margin.map(_margin =>
        s"${c.floor}\t${format(_margin, "", SHORT)}"
      )
    }
  }

  /**
    * Evaluate the single class SVM for outliers or defects detection
    */
  it should s"$name Single class model for detection of outliers" in {
    show("Single class model for detection of outliers")

    val relPath = "supervised/svm/dividends2.csv"
    val NU = 0.2
    val GAMMA = 0.5
    val EPS = 1e-3
    val NFOLDS = 2

    // Specifies the list of fundamental corporate financial metrics to
    // be extracted from the data source
    val extractor = relPriceChange ::
      debtToEquity ::
      dividendCoverage ::
      cashPerShareToPrice ::
      epsTrend ::
      dividendTrend :: List[Array[String] =>Double]()

    // filter to distinguish null and non-null values
    val filter = (x: Double) => if(x == 0) -1.0 else 1.0

    // Convert input data into a observations x features matrix
    def getObservations(input: Features): Try[Features] = Try {
      val vec = input.dropRight(1).map(_.toArray)
      transpose(vec).toVector
    }

    // Configuration specific to the One-class Support vector classifier using
    // the radius basis functions kernel
    val config = SVMConfig(
      new OneSVCFormulation(NU),
      new RbfKernel(GAMMA),
      SVMExecution(EPS, NFOLDS))

    (for {
      path <- getPath(relPath)
      pfn <- DataSource(path, true, false, 1).map(_.|>)
       // Apply the extractor
      input <- pfn(extractor)

      // Build the labeled input data
      obs <- getObservations(input)
    } yield {
      // Train the SVM model
      val svc = SVM[Double](config, obs, input.last.map(filter(_)).toVector)
      show(s"$name ${svc.toString}\naccuracy ${svc.accuracy.getOrElse(-1.0)}")
    })
    .getOrElse( error("SVCEval failed"))
  }


  /**
    * Evaluate the SVM model as a regression tool
    */
  it should s"$name Support vector regression" in {
    import org.scalaml.trading.GoogleFinancials._
    show("Support vector regression")

    val relPath = "supervised/svm/SPY.csv"
    val C = 12
    val GAMMA = 0.3
    val EPS = 1e-3
    val EPSILON = 2.5
    val NUM_DISPLAYED_VALUES = 128

    val config = SVMConfig(new SVRFormulation(C, EPSILON), new RbfKernel(GAMMA))

    def getLabeledData(numObs: Int): Try[(Features, DblVec)] = Try {
      val y = Vector.tabulate(numObs)(_.toDouble)
      val xt = Vector.tabulate(numObs)(Array[Double](_))
      (xt, y)
    }

    (for {
      path <- getPath(relPath)
      src <- DataSource(path, false, true, 1)
      price <-  src.get(close)
      (xt, y) <- getLabeledData(price.size)
      linRg <- SingleLinearRegression[Double](y, price)
    } yield {
        val svr: SVM[Double] = SVM[Double](config, xt, price)
        show(s"$name First $NUM_DISPLAYED_VALUES time series datapoints\n")
        display(s"$name vs. Linear Regression",
          collect(svr, linRg, price),
          List[String]("Support vector regression", "Linear regression", "Stock Price"))
        1
      }).get


    /**
      * Collect the data generated by the simple regression and and support vector regression models
      */
    def collect(
      svr: SVM[Double],
      lin: SingleLinearRegression[Double],
      price: Vector[Double]): List[Vector[DblPair]] = {

      import scala.language.postfixOps

      val pfSvr = svr |>
      val pfLin = lin |>

      // Create buffers to collect data from the two regression models
      val svrResults = ArrayBuffer[DblPair]()
      val linResults = ArrayBuffer[DblPair]()

      val r = Range(1, price.size - 80)
      val selectedPrice = r.map(x => (x.toDouble, price(x)))

      r.foreach( n => {
        for {
          x <- pfSvr(Array[Double](n.toDouble))
          if pfLin.isDefinedAt(n)
          y <- pfLin(n)
        } yield  {
          svrResults.append((n.toDouble, x))
          linResults.append((n.toDouble, y))
        }
      })
      show(s"$name Price\n${price.mkString(",")}")
      show(s"$name Linear Regression\n${linResults.map(_._2).mkString(",")}")
      show(s"$name Support vector regression\n${svrResults.map(_._2).mkString(",")}")

      List[Vector[DblPair]](svrResults.toVector, linResults.toVector, selectedPrice.toVector)
    }

    def display(label: String, xs: List[Vector[DblPair]], lbls: List[String]): Unit = {
      import org.scalaml.plots.{ScatterPlot, LightPlotTheme, Legend}
      require( !xs.isEmpty, s"$name Cannot display an undefined time series")

      val plotter = new ScatterPlot(Legend("SVREval", "SVM regression SPY prices", label, "SPY"), new LightPlotTheme)
      plotter.display(xs, lbls, 340, 250)
    }
  }
}


// ---------------------------------------  EOF ------------------------------------------------------
