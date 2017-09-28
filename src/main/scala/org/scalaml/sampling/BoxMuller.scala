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

import scala.util.Random


/**
  * Generic trait for the sampling against a given Gaussian distribution
  * @author Patrick Nicolas
  * @version 0.99.2
  *
  * @see Scala for Machine Learning Chapter 8, Monte Carlo Inference - Box-Muller algorithm
  */
private[scalaml] trait GaussianSampling {
  def nextDouble: Double
}


/**
  * Class that implements the Box Muller algorithm to generate a sample from
  * a Gaussian distribution cosine and sine components
  * {{{
  *      Given two uniform distribution U1, and U2
  *      Zc = sqrt(-2.logU1). cos(2.pi.U2) is a normally distributed random variable
  *      Zs = sqrt(-2.logU1). sin(2.pi.U2) is a normally distributed random variable
  *      Zc and Zs are independent
  * }}}
  * @author Patrick Nicolas
  * @version 0.99.2
  *
  * @param r Uniform distribution variable over interval [0, 1]. The Scala uniform random generator
  *          is used as a default uniform distribution
  * @param cosine Flag to generate the cosine distribution component Zc if true, the sine
  *               distribution Zs otherwise.
  * @see Scala for Machine Learning Chapter 8, Monte Carlo Inference - Box-Muller algorithm
  */
final private[scalaml] class BoxMuller(
  r: () => Double = () => Random.nextDouble,
  cosine: Boolean = true)
extends GaussianSampling {
  import Math._

    // Pseudo-normalized uniform distribution
  private def uniform2PI: Double = 2.0*PI*r()

  /**
    * Generate a sample value from a normal distribution using Zc formula if cosine flag is
    * true, Zs formula is cosine flag is false
    * @return Gaussian sampled value
    */
  override def nextDouble: Double = {
    val theta = uniform2PI
    val x = -2.0*log(r())
    sqrt(x) * (if(cosine) cos(theta) else sin(theta))
  }
}

// -------------------  EOF ------------------------------------------------