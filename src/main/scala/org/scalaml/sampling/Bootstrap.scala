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
package org.scalaml.sampling

import scala.collection.mutable
import scala.util.Random

/**
  * Definition of the Bootstrapping method with replacement
  * {{{
  *
  * }}}
  * @author Patrick Nicolas
  * @version 0.99.2
  * @todo Add description of algorithm
  * @param numSamples  number of samples extracted from the original dataset
  * @param s  dispersion function
  * @param inputDistribution  Original dataset
  * @param randomizer Uniform randomization function
  * @see cala for Machine Learning Chapter 8, Monte Carlo Inference - Bootstrapping with replacement
  */
private[scalaml] final class Bootstrap (
    numSamples: Int,
    s: Vector[Double] => Double,
    inputDistribution: Vector[Double],
    randomizer: Int => Int
  ) {

    lazy val bootstrappedReplicates: Array[Double] =
      (0 until numSamples)./:(mutable.ArrayBuffer[Double]())(
        (buf, _) => buf += createBootstrapSample
      ).toArray

    def createBootstrapSample: Double = s(
      (0 until inputDistribution.size)./:(mutable.ArrayBuffer[Double]())(
        (buf, _) => {
          val randomValueIndex = randomizer( inputDistribution.size )
          buf += inputDistribution( randomValueIndex )
        }
      ).toVector
    )

    lazy val mean = bootstrappedReplicates.reduce( _ + _ )/numSamples

    final def error: Double = {
      import Math._
      val sumOfSquaredDiff = bootstrappedReplicates.reduce(
        (s1: Double, s2: Double) =>
          (s1 - mean)*(s1 - mean) +  (s2 - mean)*(s2 - mean)
      )
      sqrt(sumOfSquaredDiff / (numSamples - 1))
  }
}


// ----------------------------  EOF -------------------------------------------------