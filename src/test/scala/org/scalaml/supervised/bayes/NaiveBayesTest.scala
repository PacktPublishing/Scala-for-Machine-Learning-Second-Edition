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
package org.scalaml.supervised.bayes

import java.text.SimpleDateFormat

import org.scalaml.{Logging, Resource}
import org.scalaml.Predef.{DblPair, DblVec}
import org.scalaml.filtering.movaverage.SimpleMovingAverage
import org.scalaml.validation.OneFoldValidation
import org.scalaml.workflow.data.{DataSource, DocumentsSource}
import org.scalaml.stats.{Difference, Transpose}
import org.scalatest.{FlatSpec, Matchers}
import Difference._
import Transpose._
import org.scalaml.supervised.bayes.text.{Lexicon, TextAnalyzer}

import scala.util.Try
import org.scalaml.trading.YahooFinancials._

import scala.collection.{Seq, Set, immutable, mutable}

trait BayesTest extends Resource {
  protected val relPath = "supervised/bayes/"

  /**
   * Extractor for data in files associated to a list of ticker symbols
   */
  protected val extractor = toDouble(CLOSE) ::
    ratio(HIGH, LOW) ::
    toDouble(VOLUME) ::
    List[Array[String] => Double]()
  protected def symbolFiles = DataSource.listSymbolFiles(getPath(relPath).get)
}


/**
  * Unit test for Binary Naive Bayes model
  */
final class NaiveBayesTest extends FlatSpec with Matchers with Logging with BayesTest {
  protected[this] val name = "Naive Bayes"

  type Output = List[Array[Int]]
  type Obs = Vector[Array[Double]]
  type Labeled = (Obs, Vector[Array[Int]])
  final private val TRAIN_VALIDATION_RATIO = "0.8"

  it should s"$name Binomial Naive Bayes for IBM period=8" in {
    show("$name Binomial Naive Bayes for IBM period=8")
    validate(Array[String]("IBM", TRAIN_VALIDATION_RATIO, "8"))
  }

  it should s"$name Binomial Naive Bayes for NEM period=4" in {
    show("$name Binomial Naive Bayes for NEM period=4")
    validate(Array[String]("NEM", TRAIN_VALIDATION_RATIO, "4"))
  }

  it should s"$name Binomial Naive Bayes for NEM period=12" in {
    show("$name Binomial Naive Bayes for NEM period=12")
    validate(Array[String]("NEM", TRAIN_VALIDATION_RATIO, "12"))
  }

  it should s"$name Binomial Naive Bayes forvNEM period=36" in {
    show(s"$name Binomial Naive Bayes forvNEM period=36")
    validate(Array[String]("NEM", TRAIN_VALIDATION_RATIO, "36"))
  }

  it should s"$name Binomial Naive Bayes for GE period=4" in {
    show(s"$name Binomial Naive Bayes for GE period=4")
    validate(Array[String]("GE", TRAIN_VALIDATION_RATIO, "4"))
  }

  it should s"$name Binomial Naive Bayes forvGE period=12" in {
    show(s"$name Binomial Naive Bayes forvGE period=12")
    validate(Array[String]("GE", TRAIN_VALIDATION_RATIO, "12"))
  }

  it should s"$name Binomial Naive Bayes for GE period=36" in {
    show("$name Binomial Naive Bayes for GE period=36")
    validate(Array[String]("GE", TRAIN_VALIDATION_RATIO, "36"))
  }

  it should s"$name Binomial Naive Bayes for BAC period=4" in {
    show("$name Binomial Naive Bayes for BAC period=4")
    validate(Array[String]("BAC", TRAIN_VALIDATION_RATIO, "4"))
  }

  it should s"$name Binomial Naive Bayes for BAC period=12" in {
    show("$name Binomial Naive Bayes for BAC period=12")
    validate(Array[String]("BAC", TRAIN_VALIDATION_RATIO, "12"))
  }

  it should s"$name Binomial Naive Bayes for BAC period=36" in {
    show("$name Binomial Naive Bayes for BAC period=36")
    validate(Array[String]("BAC", TRAIN_VALIDATION_RATIO, "36"))
  }

  private def validate(args: Array[String]) {
    val symbol = s"${args(0)}.csv"
    val trainRatio = args(1).toDouble
    val period = args(2).toInt
    val description = s"symbol:${args(0)} smoothing period:$period"

    // Partial function for the simple moving average
    val pfnMv = SimpleMovingAverage[Double](period, false) |>

    // Binary quantization of the difference between two values in a pair
    val delta = (x: DblPair) => if (x._1 > x._2) 1 else 0

    // Compute the difference between the value of the features of an observation, obs(n) and
    // its simple moving average sm(i) for all i.
    def computeDeltas(obs: Obs): Try[Labeled] = Try {

      // Simple moving average value
      val sm = obs.map(_.toVector).map(pfnMv(_).getOrElse(Vector.empty[Double]).toArray)
      // Observations beyond the initial period of observations
      val x = obs.map(_.drop(period - 1))
      // Compute the difference
      (x, x.zip(sm).map { case (_x, y) => _x.zip(y).map(delta(_)) })
    }

    (for {
      path <- getPath(relPath)
      // Partial function for the extraction of data
      pfnSrc <- DataSource(symbol, path, true, 1).map(_.|>)
      // Extracts the observations
      obs <- pfnSrc(extractor)

      // Compute the difference between the value of a feature and its moving average
      (x, deltas) <- computeDeltas(obs)

      // Extract the labels or expected outcome using the Difference
      expected <- Try {
        difference(x.head.toVector, diffInt)
      }

      // Extract the time series of features using Transpose
      features <- Try {
        transpose(deltas)
      }

      // Applies a one fold validation of the features and expected values using
      // a ratio of the size of the training set, trainRatio
      labeledData <- OneFoldValidation[Int](features.tail, expected, trainRatio)

      // Constructs the Naive Bayes model and Computes the F1 score
      f1Score <- {
        NaiveBayes[Int](1.0, labeledData.trainingSet).validate(labeledData.validationSet)
      }
    } yield {
      show(s"$name Time series of observations\n")
      show(obs.take(64).map(_.mkString("\n")).mkString("\n"))

      show(s"$name Model input features\n")
      show(features.take(64).map(_.mkString(",")).mkString("\n"))

      val labels = Array[String](
        "price/ave price",
        "volatility/ave. volatility",
        "volume/ave. volume"
      )
    }).getOrElse(-1)
  }

  val pathCorpus = "supervised/bayes/input"
  val pathLexicon = "supervised/bayes/lexicon.txt"

  it should s"$name text analysis" in {

    val dateFormat: SimpleDateFormat = new SimpleDateFormat("MM.dd.yyyy")

    // Map of keywords which associates keywords with a semantic equivalent (poor man's stemming
    // and lemmatization), loaded from a dictionary/lexicon file
    val LEXICON = loadLexicon

    // Regular expression to extract keywords and match them against the lexicon
    def parse(content: String): Array[String] = {
      val regExpr = "['|,|.|?|!|:|\"]"
      content.trim.toLowerCase
        .replace(regExpr, " ")
        .split(" ")
        .filter(_.length > 2)
    }

    // Stock prices for TSLA indexed by their date from the oldest to the newest.
    val TSLA_QUOTES = Vector[Double](
      250.56, 254.84, 252.66, 252.94, 253.21, 255.84, 234.41, 241.49, 237.79, 230.97,
      233.98, 240.04, 235.84, 234.91, 228.89, 220.17, 220.44, 212.96, 207.32, 212.37,
      208.45, 216.97, 230.29, 225.4, 212.22, 207.52, 215.46, 203.93, 204.19, 197.78,
      198.09, 201.91, 199.11, 198.12, 197.38, 205.64, 207.99, 209.86, 199.85
    )

    import scala.language.postfixOps

    // Generate the partial function to extract documents from a directory or set of files
    val pfnDocs = DocumentsSource(dateFormat, getTextPath(pathCorpus).get) |>

    // Create a text analyzer with a time stamp of type Long, a parsing function and a
    // given Lexicon
    val textAnalyzer = TextAnalyzer[Long](parse, Lexicon(LEXICON))

    // Partial function for the text analyzer
    val pfnText = textAnalyzer |>

    (for {
      // Extract the corpus from the documents
      corpus <- pfnDocs(None)

      // Extract the terms frequencies map from the corpus
      if pfnText.isDefinedAt(corpus)
      termsFreq <- pfnText(corpus)

      // Create the features set from quantization
      featuresSet <- textAnalyzer.quantize(termsFreq)

      // Extract the labels by computing the difference of the stock prices
      // between trading sessions
      expected <- Try { difference(TSLA_QUOTES, diffInt) }
    } yield {
      val naiveBayes = NaiveBayes[Double](1.0, featuresSet._2.zip(expected))
      val likelihoods = naiveBayes.model.get.likelihoods
      val means: Seq[DblVec] = likelihoods.map(_.muSigma.map(_._1))
      display(means, s"$name text classification", Set[String]("Positives", "Negatives"))

      val results =
        s"""Corpus ----- \n${corpus.mkString("\n\n")}
           | Terms frequencies ----- \n
           | ${termsFreq.map(_.map(e => s"(${e._1},${e._2})").mkString(" | ")).mkString("\n")}
           | Expected ----- \n${expected.mkString("\n")}
           | Quantized ----- \n${featuresSet._1.mkString(",")}
           | \n${featuresSet._2.map(_.mkString(",")).mkString("\n")}
           | \nText extraction model${naiveBayes.toString(featuresSet._1)}""".stripMargin
      show(s"$name $results")
    }).getOrElse(-1)
  }

  /**
    * Extract the lexicon (HashMap) from a file.
    */
  private def loadLexicon: immutable.Map[String, String] = {
    import scala.io.Source._

    val src = fromFile(getTextPath(pathLexicon).get)
    val fields: Array[Array[String]] = src.getLines.map(_.split(",").map(_.trim)).toArray
    val lexicon = fields./:(mutable.HashMap[String, String]())((hm, field) => {
      show(s"${field(0)} -> ${field(1)}")
      hm += ((field(0), field(1)))
    }
    ).toMap

    src.close
    lexicon
  }

  private def display(values: Seq[DblVec], label: String, labels: Set[String]): Unit = {
    import org.scalaml.plots.{LinePlot, LightPlotTheme, Legend}

    val legend = Legend(
      name, label, "Positive/negative outcome", "Mean values"
    )

    val dataPoints = values.map(_.toVector).zip(labels)
    LinePlot.display(dataPoints.toList, legend, new LightPlotTheme)
  }
}


// ---------------------------------  EOF ----------------------------------------------