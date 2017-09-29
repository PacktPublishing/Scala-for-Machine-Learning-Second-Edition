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

import org.scalatest.{FlatSpec, Matchers}
import scala.util.Random

/**
  * Unit test for the MCMC, Metropolis-Hastings algorithm for a single-variable feature
  */
final class MetropolisHastingsTest extends FlatSpec with Matchers with org.scalaml.Logging {
  protected[this] val name = "MCMC Metropolis-Hastings"

  val square = (x: Double) => if(x < 0.0 && x >= 1.0) 0.0 else x
  val linear = (x: Double) => 2.0*x -1.0

  it should s"$name evaluation square signal with 20 iterations and 0.5 initial value" in {
    show(s"Evaluation square signal with 20 iterations and 0.5 initial value")

    val numIterations = 20
    val initialValue = 0.5

    val results = test(numIterations, initialValue)
    val acceptance = results.acceptedRate(numIterations)
    acceptance > 0.80 should be (true)
    show(s"$name ${results.toString}\n$acceptance")
  }


  it should s"$name evaluation square signal with 100 iterations and 0.5 initial value" in {
    show("Evaluation square signal with 100 iterations and 0.5 initial value")

    val numIterations = 100
    val initialValue = 0.5

    val results = test(numIterations, initialValue)

    val acceptance = results.acceptedRate(numIterations)
    acceptance > 0.80 should be (true)
    show(s"$name ${results.toString}\n$acceptance")
  }

  it should s"$name evaluation square signal with 250 iterations and 0.5 initial value" in {
    show("Evaluation square signal with 100 iterations and 0.5 initial value")

    val numIterations = 250
    val initialValue = 0.5

    val results = test(numIterations, initialValue)
    val acceptance = results.acceptedRate(numIterations)
    acceptance > 0.80 should be (true)
    show(s"$name ${results.toString}\n$acceptance")
  }

  it should s"$name evaluation square signal with 250 iterations and 1.0 initial value" in {
    show("Evaluation square signal with 250 iterations and 1.0 initial value")

    val numIterations = 250
    val initialValue = 1.0

    val results = test(numIterations, initialValue)
    val acceptance = results.acceptedRate(numIterations)
    acceptance > 0.80 should be (true)
    show(s"$name ${results.toString}\n$acceptance")
  }

   private def test(numIters: Int, initialValue: Double): Trace = {
    val random = new Random
    val q = (s: Double, sPrime: Double) => 0.5*(s + sPrime)
    val proposer = (s: Double) => {
      val r = random.nextDouble
      (if(r < 0.2 || r > 0.8) s*r else 1.0)
    }

    val mh = new OneMetropolisHastings(square, q, proposer, ()=>random.nextDouble)
    mh.mcmc(initialValue, numIters)
  }
}


// ----------------------------  EOF -----------------------------------------------