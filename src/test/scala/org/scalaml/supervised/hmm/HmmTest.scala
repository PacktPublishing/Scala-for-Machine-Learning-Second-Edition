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
package org.scalaml.supervised.hmm

import org.scalaml.Logging
import org.scalaml.util.MathUtils.DMatrix
import org.scalaml.stats.TSeries
import org.scalatest.{FlatSpec, Matchers}


final class HmmTest extends FlatSpec with Matchers with Logging {
  protected val name: String = "Hidden Markov Model"

  it should s"$name decoder 2x2 using given dictionary (symbols)" in {
    import HMMModel._, HMM._
    show("$name decoder 2x2 using given dictionary (symbols)")

    // Quantization function
    implicit def quantize(xt: Array[Double]): Int = xt.head.toInt
    val symbols = Map[Char, Int]('A' -> 0, 'B' -> 1, 'C' -> 2, 'D' ->3)

      // Step 1: Define the lambda model
      // State-transition probabilities matrix for HMM
    val A0 = Array[Array[Double]](Array[Double](0.4, 0.6), Array[Double](0.5, 0.5))

      // Emission/observations probabilities matrix for
      //[A, B, C, D]
    val B0 =  Array[Array[Double]](
        // C, A, D, D
      Array[Double](0.0, 0.0, 0.25, 0.75),
        // A, B, B, B
      Array[Double](0.25, 0.75, 0.0, 0.0)
    )
    val PI0 = Array[Double](0.6, 0.4)

    if( !validate(A0, B0, PI0))
      throw new IllegalStateException(s"$name incorrect lambda model")
    show(s"$name Lambda model validated")

      // Step 2: Specifies and normalized the sequence of observations
    val observed = Vector[Char]('D', 'C', 'D', 'D', 'C', 'C', 'A', 'A', 'B', 'B', 'B', 'D', 'A')
    val xt = observed.flatMap( symbols.get(_)).map(Array[Double](_))
    show(s"$name Input ${xt.map(_.mkString).mkString(",")}")

      // Step 3: Decode the sequence of states using Viterbi algorithm
    val lambda = HMMModel(DMatrix(A0), DMatrix(B0), PI0, xt.length)
    val hmmPredictor = decode(lambda, xt).map( _.toString)


    hmmPredictor.isDefined should be (true)
    hmmPredictor.map( show(_)).getOrElse(error(s"$name decoding failed"))
  }


  /**
    * Unit test for decoding mode of HMM
    */
  it should s"$name decoder 3x3" in {
    import HMMModel._, HMM._, TSeries._
    show(s"$name decoder 3x3")

    implicit def quantize(xt: Array[Double]): Int = if(xt.head > 0.5) 1 else 0

    // Step 1: Define the lambda model
    // State-transition probabilities matrix for HMM
    val A0 = Array[Array[Double]](
      Array[Double](0.4, 0.3, 0.3),
      Array[Double](0.5, 0.3, 0.2),
      Array[Double](0.4, 0.6, 0.0)
    )

    // Emission/observations probabilities matrix
    val B0 =  Array[Array[Double]](
      Array[Double](0.3, 0.7),
      Array[Double](0.7, 0.3),
      Array[Double](0.8, 0.2)
    )

    val PI0 = Array[Double](0.3, 0.4, 0.3)

    if( !validate(A0, B0, PI0))
      throw new IllegalStateException(s"$name incorrect lambda model")
    show(s"$name: Lambda model validated")

    // Step 2: Specifies the sequence of observations
    val observed = Vector[Double](
      1.0, 2.0, 8.9, 13.3, 11.1, 9.5, 0.5, 0.3, 0.0, 0.8, 0.1, 2.6, 4.7, 10.8, 0.7, 1.8, 3.9,
      6.0, 5.2, 4.7, 6.0, 4.9, 5.7
    )

    val yt = normalize(observed).get.map( Array[Double](_) )

    // Step 3: Decode the sequence of states using Viterbi algorithm
    val lambda = HMMModel(DMatrix(A0), DMatrix(B0), PI0, yt.length)
    val hmmPredictor = decode(lambda, yt).map( _.toString)
    hmmPredictor.isDefined should be (true)
    hmmPredictor.map( show(_)).getOrElse(error(s"$name decoding failed"))
  }

  /**
    * Unit test for the evaluation of HMM
    */
  it should s"$name HMM evaluation" in {
    import scala.language.postfixOps
    import HMMModel._, HMM._
    show(s"$name HMM evaluation" )
    val EPS = 1e-2

    // Step 1. Defined the Lambda model
    // State-transition probabilities matrix for HMM
    val A0 = Array[Array[Double]](
      Array[Double](0.21, 0.13, 0.25, 0.06, 0.11, 0.24),
      Array[Double](0.31, 0.17, 0.18, 0.04, 0.19, 0.11),
      Array[Double](0.32, 0.15, 0.15, 0.01, 0.21, 0.16),
      Array[Double](0.25, 0.12, 0.11, 0.01, 0.27, 0.24),
      Array[Double](0.22, 0.10, 0.09, 0.03, 0.31, 0.25),
      Array[Double](0.15, 0.08, 0.05, 0.07, 0.43, 0.22)
    )

    // Emission/observations probabilities matrix
    val B0 =  Array[Array[Double]](
      Array[Double](0.45, 0.12, 0.43),
      Array[Double](0.54, 0.26, 0.20),
      Array[Double](0.25, 0.51, 0.24),
      Array[Double](0.46, 0.33, 0.21),
      Array[Double](0.33, 0.57, 0.10),
      Array[Double](0.42, 0.39, 0.19)
    )

    val PI0 = Array[Double](0.26, 0.04, 0.11, 0.26, 0.19, 0.14)
    if( !validate(A0, B0, PI0))
      throw new IllegalStateException(s"$name incorrect lambda model")

      // Step 2: Defined the sequence of observed states
    val data = Vector[Double](
      0.0, 1.0, 2.0, 1.0, 3.0, 0.0, 1.0, 2.0, 2.0, 1.0, 3.0, 0.0, 1.0, 0.0, 2.0, 1.0
    )

      // The quantization method consists of normalizing the input data over the
      // number (or range) of symbols associated to the lambda model. The number of symbols
      // is the size of the rows of the emission probabilities matrix B
    val max = data.max
    val min = data.min
    implicit val quantize = (x: Array[Double]) => ((x.head/(max - min) + min)*(B0.head.length-1)).toInt

      // Step 3: Create a model
    val xt = data.map(Array[Double](_))
    val lambda = HMMModel(DMatrix(A0), DMatrix(B0), PI0, data.length)

    val validated = lambda.validate(EPS)
    validated should be (true)

      // Step 4: Compute the likelihood of the sequence of observations
      // Make sure the values in the input lambda model are normalized
    if( validated )
      // Invokes the evaluation form for the lambda model
      evaluate(lambda, xt).map( _.toString).map( show(_)).getOrElse(error(s"$name evaluation failed"))
    else
      error(s"$name Lambda model for evaluation is not properly formed")
  }

  /**
    * Unit test for the Baum-Welch Expectation-Maximization
    */
  it should s"$name training" in {
    show(s"$name training")
    val CSV_DELIM= ","
    val NUM_SYMBOLS = 6
    val NUM_STATES = 5
    val EPS = 1e-4
    val MAX_ITERS = 150

    // Step 1: Specifies the sequence of observations
    val observations = Vector[Double](
      0.01, 0.72, 0.78, 0.56, 0.61, 0.56, 0.45, 0.42, 0.46, 0.38, 0.35, 0.31, 0.32, 0.34,
      0.29, 0.23, 0.21, 0.24, 0.18, 0.15, 0.11, 0.08, 0.10, 0.03, 0.00, 0.06, 0.09, 0.13,
      0.11, 0.17, 0.22, 0.18, 0.25, 0.30, 0.29, 0.36, 0.37, 0.39, 0.38, 0.42, 0.46, 0.43,
      0.47, 0.50, 0.56, 0.52, 0.53, 0.55, 0.57, 0.60, 0.62, 0.65, 0.68, 0.65, 0.69, 0.72,
      0.76, 0.82, 0.87, 0.83, 0.90, 0.88, 0.93, 0.92, 0.97, 0.99, 0.95, 0.91
    )

    implicit val quantize = (x: Array[Double]) => (x.head* (NUM_STATES+1)).floor.toInt

    val xt: Vector[Array[Double]] = observations.map(Array[Double](_))
    val config = HMMConfig(xt.size, NUM_STATES, NUM_SYMBOLS, MAX_ITERS, EPS)

    // Step 2 Extract the HMM model hmm.model
    val hmm = HMM[Double](config, xt, EVALUATION())
    hmm.isModel should  be (true)
    if( hmm.isModel)
      show(s"$name Lambda model training is valid:\n${hmm.toString}")
    else
      error(s"$name Training failed")
  }
}


// -----------------------------------  EOF -------------------------------------------
