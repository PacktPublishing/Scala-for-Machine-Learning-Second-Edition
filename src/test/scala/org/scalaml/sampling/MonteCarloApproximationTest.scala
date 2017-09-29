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

import org.scalaml.Logging
import org.scalatest.{FlatSpec, Matchers}


/**
  * Unit test for the Monte Carlo numerical integration
  */
final class MonteCarloApproximationTest extends FlatSpec with Matchers with Logging {
  import Math._
  protected val name = "Monte Carlo Approximation"
  private final val NumPoints = 50000

  /**
    * Integration of
    */
  it should s"$name of f(x) = sqrt(1+x)" in {
    show("$name of f(x) = sqrt(1+x)")

    val approximator = new MonteCarloApproximation((x: Double) => sqrt(1.0+x), NumPoints)
    val expected = 1.5735
    abs(approximator.sum(1.0, 2.0) - expected) < 0.01 should be (true)
  }

  it should s"$name of f(x) = 1/x" in {
    show(s"$name of f(x) = 1/x")

    val approximator = new MonteCarloApproximation((x: Double) => 1.0/x, NumPoints)

    val volume = approximator.sum(1.0, 2.0)
    val expected = log(2.0) - log(1.0)
    abs(approximator.sum(1.0, 2.0) - expected) < 0.01 should be (true)
  }

}


// -------------------------  EOF -----------------------------------------------