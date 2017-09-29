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
package org.scalaml.unsupervised.divergence

import org.apache.commons.math3.distribution.GammaDistribution
import org.scalaml.Logging
import org.scalatest.{FlatSpec, Matchers}

import scala.util.Random

/**
 *
 */
final class KullbackLeiblerTest extends FlatSpec with Matchers with Logging {
  protected[this] val name = "Kullback Leibler divergence"

  it should s"$name Kullback Leibler test on two data sets" in {
    show("$name Kullback Leibler test on two data sets")

    val numDataPoints = 100000
    def gammaDistribution( shape: Double, scale: Double ): Seq[Double] = {
      val gamma = new GammaDistribution( shape, scale )
      Seq.tabulate( numDataPoints )( n => gamma.density( 2.0 * Random.nextDouble ) )
    }

    val kl = new KullbackLeibler[Double]( gammaDistribution( 2.0, 1.0 ), gammaDistribution( 2.0, 1.0 ) )
    val divergence = kl.divergence( 100 )
    val expectedDivergence = 0.0063
    Math.abs( divergence - expectedDivergence ) < 0.001 should be( true )
    show( s"$name divergence $divergence" )

    val kl2 = new KullbackLeibler[Double]( gammaDistribution( 2.0, 1.0 ), gammaDistribution( 1.0, 0.5 ) )
    val divergence2 = kl2.divergence( 100 )

    val expectedDivergence2 = 2.655
    Math.abs( divergence2 - expectedDivergence2 ) < 0.1 should be( true )
    show( s"$name divergence $divergence2" )
  }
}


// -------------------------------------------  EIF ----------------------------------------------
