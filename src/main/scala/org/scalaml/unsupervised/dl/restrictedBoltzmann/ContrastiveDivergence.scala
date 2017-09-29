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

import org.scalaml.Predef._
import org.scalaml.Predef.Context._
import org.apache.commons.math3.linear.Array2DRowRealMatrix
import org.apache.commons.math3.linear.RealMatrix
import RBM._

import scala.annotation.implicitNotFound



/**
  * Implementation of the Contrastive Divergence to approximate the computation of the logarithm of the
  * gradient of the likelihood.
  * The computation of the contrastive divergence relies on the matrix multiplication
  * method defined in Apache Commons Math.
  * @author Patrick Nicolas
  * @version 0.99.2
  * @param nSamples Size of the sample to be extracted from the condition probability
  * @tparam T Type contextual bound to a Double
  * @see Scala for Machine Learning Chapter 11 Deep learning/ Restricted Boltzmann Machine
  */
@throws(classOf[IllegalArgumentException])
@implicitNotFound(msg = "Contrastive divergence implicit conversion to Double undefined")
private[scalaml] class ContrastiveDivergence[@specialized(Double) T: ToDouble](nSamples: Int) {
  require(nSamples > 1, s"Cannot compute contrastive divergence with $nSamples samples")

  /**
    * Method to compute the contrastive divergence using a sampler and conditional
    * probability of latent features given an observed value and vice-versa.
    * @param weights Weights for the connectivity between input layer (observations) and hidden
    *                layer (latent features).
    * @param v0  Initial set of observations
    * @return A matrix of type
    */
  def apply(weights: Weights, v0: Array[Array[T]]): RealMatrix = {
      // Type conversion
    val cV0 = v0.map(_.map(implicitly[ToDouble[T]].apply(_)))
    val probSampler = new SCondProb(weights, nSamples)

      // Sample h0 = p(v0|h)
    val h0 = cV0.map( probSampler.sample(_, true) )
      // positive energy
    val posEnergy = multiplyTranspose(h0, cV0)

      // Sample v1 = p(h0| v)
    val v1 = h0.map( probSampler.sample(_, false))
      // Sample h1 = p(v1| h)
    val h1 = v1.map( probSampler.sample(_, true))
      // Negative energy
    val negEnergy = multiplyTranspose(v1, h1)
      // Normalized difference in energy
    posEnergy.subtract(negEnergy).scalarMultiply(v0.head.length)
  }


  private def multiplyTranspose(input1: DblMatrix, input2: DblMatrix): RealMatrix = {
    val realMatrix1 = new Array2DRowRealMatrix(input1)
    val realMatrix2 = new Array2DRowRealMatrix(input2)
    realMatrix1.transpose.multiply(realMatrix2)
  }
}


// ------------------------  EOF ----------------------------------------------