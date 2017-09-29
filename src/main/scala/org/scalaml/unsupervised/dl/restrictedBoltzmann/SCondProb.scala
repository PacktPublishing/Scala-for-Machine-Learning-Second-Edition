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

import org.scalaml.stats.TSeries.innerDouble
import org.scalaml.util.MathUtils.sigmoid
import RBM._


/**
  * Computation of the sampling of likelihood using an activation function
  * @author Patrick Nicolas
  * @version 0.99.2
  * @param weights Connection weights
  * @param sampleSize Number of samples extracted from conditional probabilities
  * @see Scala for Machine Learning Chapter 11 Deep learning/ Restricted Boltzmann Machine
  */
private[scalaml] final class SCondProb(weights: Weights, sampleSize: Int) {

  /**
    * Compute the raw conditional probability p(h|v)
    * @param h input for the conditional probability
    */
  @throws(classOf[IllegalArgumentException])
  def probHV(h: Array[Double]): IndexedSeq[Double] = {
    require(h.size == weights.size -1, "Incorrect number of hidden nodes")

    (0 until h.size).map(n =>
      sigmoid(innerDouble(h, weights(n+1)) + weights(0)(n)) - weights(0)(n)
    )
  }

  /**
    * Compute the raw conditional probability p(v|h)
    */
  @throws(classOf[IllegalStateException])
  def probVH(v: Array[Double]): IndexedSeq[Double] = {
      // Transpose the original matrix of weights.
    val tWeights = weights.transpose
    require(v.size == tWeights.size -1, "Incorrect number of input/visible nodes")

    (0 until v.size).map(n =>
      sigmoid(innerDouble(v, tWeights(n+1)) + tWeights(0).head) - tWeights(0)(n)
    )
  }

  /**
    * Compute the sampling of a given conditional probability
    * @param input Data input for either the input/data layer or the hidden layer
    * @param positive Flag to select p(h|v) if true, p(v|h) otherwise
    * @return
    */
  def sample(input: Array[Double], positive: Boolean): Array[Double] = {
    import scala.util.Random._

    val rawOut = if(positive) probHV(input) else probVH(input)
    (0 until sampleSize).map( _ => rawOut(nextInt(input.length))).toArray
  }
}


// ----------------------------   EOF --------------------------------------------------
