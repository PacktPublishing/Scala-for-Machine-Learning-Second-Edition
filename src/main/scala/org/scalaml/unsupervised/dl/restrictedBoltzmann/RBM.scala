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
package org.scalaml.unsupervised.dl.restrictedBoltzmann

import org.apache.commons.math3.linear.{Array2DRowRealMatrix, RealMatrix}
import org.scalaml.Predef.Context.ToDouble
import org.scalaml.core.ITransform

import scala.util.Random._
import scala.util.Try

/**
  * Definition of the binary restricted Boltzmann machine
  * @author Patrick Nicolas
  * @version 0.99.2
  * @param numInputs  Number of features or input neurons
  * @param numLatents  Size of the representation layer
  * @param config Configuration of the Restricted Boltzmann Machine
  * @param nSamples Number of samples to be generated for each of the conditional probability
  * @tparam T  Type contextually bound to a Double type
  * @see Scala for Machine Learning Chapter 11 Deep learning/ Restricted Boltzmann Machine
  */
@throws(classOf[IllegalArgumentException])
private[scalaml] class RBM[@specialized(Double) T: ToDouble](
    numInputs: Int,
    numLatents: Int,
    config: RBMConfig,
    nSamples: Int,
    input: Array[Array[T]]) extends ITransform[Array[T], Array[Double]] {
  import org.scalaml.stats.Stats._, RBM._

  require(input.size > 1, "Cannot execute RBM on undefined dataset")
  require(numInputs > 0, s"Cannot execute RBM with $numInputs inputs")
  require(numLatents > 0, s"Cannot execute RBM with $numLatents latent/hidden variables")

  @inline
  final def isModel: Boolean = model.isDefined

  override def |> : PartialFunction[Array[T], Try[Array[Double]]] = {
    case x: Array[T] if isModel && x.length == input.head.length =>
      val z = x.map(implicitly[ToDouble[T]].apply(_))
      val probSampler = new SCondProb(model.get, nSamples)
      Try( probSampler.probHV(z).toArray )
  }

  /**
    * Random Gaussian initialization of the bias
    */
  private[this] val model: Option[Weights] = train
  private[this] val cd = new ContrastiveDivergence[T](nSamples)

  private def train: Option[Weights] = {
    var weights: Weights = Array.fill(numInputs)(
      Array.fill(numLatents)(gauss(0.0, 0.001, nextDouble))
    )

    @scala.annotation.tailrec
    def train(x: Array[Array[T]], prevDW: RealMatrix, count: Int): Option[Weights] = {
        // Compute the divergence
      val divergence = cd(weights, x)
      val diffFactor = divergence.subtract(prevDW).scalarMultiply(config.loss)
      val dWeights = diffFactor.scalarMultiply(config.learningRate)
        // Update the weights without the
      weights = prevDW.add(dWeights).getData
        // Test for the least square error
      if(mse(prevDW, dWeights) < config.tol)
        Some(weights)
        // Test of the number of iterations has been reached
      else if(count > config.maxIters)
        None
      else
        train(input, dWeights, count+1)
    }
    train(input, new Array2DRowRealMatrix(weights), 0)
  }

  private def mse(pDW: RealMatrix, dW: RealMatrix): Double =
    pDW.subtract(dW).getData.map( _.reduceLeft(_*_)).sum
}

/**
  *
  */
private[scalaml] object RBM {
  type Weights = Array[Array[Double]]
}


// --------------------------   EOF --------------------------------------------------------
