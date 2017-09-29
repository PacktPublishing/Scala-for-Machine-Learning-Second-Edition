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
package org.scalaml.stats

import org.scalaml.Logging
import org.scalaml.util.Assertable
import org.scalatest.{FlatSpec, Matchers}


/**
  * Simple unit tests for the different computation of loss functions
  */
final class LossTest extends FlatSpec with Matchers with Logging with Assertable {
  protected[this] val name = "Loss"

  it should s"$name cross entropy arrays 1" in {
    show(s"$name cross entropy arrays 1")
    val x = Array[Double](2.0, 4.0, 6.0)
    val y = Array[Double](3.0, 6.0, 9.0)

    val xEntropy = Loss.crossEntropy(x, y)
    xEntropy should be (-26.5 +- 0.2)
  }

  it should s"$name cross entropy arrays 2" in {
    show(s"$name cross entropy arrays 2")
    val x = Array[Double](2.0, 4.0, 6.0)
    val y = Array[Double](9.0, 6.0, 3.0)

    val xEntropy = Loss.crossEntropy(x, y)
    xEntropy should be (-20.0 +- 0.2)
  }

  it should s"$name sse" in {
    show(s"$name sse")
    val x = Array[Double](2.0, 4.0, 6.0)
    val y = Array[Double](9.0, 6.0, 3.0)

    Loss.sse(x, y) should be (7.88 +- 0.10)
  }

  it should s"$name mse" in {
    show(s"$name mse")
    val x = Array[Double](2.0, 4.0, 6.0)
    val y = Array[Double](9.0, 6.0, 3.0)

    Loss.mse(x, y) should be (4.55 +- 0.10)
  }
}

// --------------------------  EOF ----------------------------------
