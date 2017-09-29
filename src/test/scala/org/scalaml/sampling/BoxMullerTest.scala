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
  * Unit test for the Box-Muller sampling generator
  */
final class BoxMullerTest extends FlatSpec with Matchers with Logging {
  protected val name = "Box-Muller sampler"

  it should s"$name generation of Gaussian distribution" in {
    show(s"$name generation of Gaussian distribution")

    val bm = new BoxMuller()

    val len = 100000
    val distribution = Array.fill(len)( bm.nextDouble )
    val mean = distribution.reduce(_ + _)/distribution.length
    show(s"$name mean $mean")
    Math.abs(mean) < 0.01 should be (true)

    val stdDev = distribution./:(0.0)( (s, x) =>  s + (x - mean)*(x -mean))/distribution.length
    show(s"$name standard deviation $stdDev")
    Math.abs(stdDev - 1.0) < 0.01 should be (true)
  }

}


// ---------------------------  EOF ------------------------------------------------