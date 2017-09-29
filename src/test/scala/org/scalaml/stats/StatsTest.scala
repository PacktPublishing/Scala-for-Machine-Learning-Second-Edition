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

final class StatsTest extends FlatSpec with Matchers with Logging with Assertable {
  protected[this] val name = "Basic statistics"
  final private val INPUT = Vector[Double](1.0, 0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5)

  it should s"$name mean & standard deviation" in {
    show(s"$name mean & standard deviation")
    val stats = Stats[Double](INPUT)

    stats.mean should be (1.125 +- 0.01)
    stats.stdDev should be (0.834 +- 0.01)
  }

  it should s"$name Laplace smoothing" in {
    show(s"$name Laplace smoothing")
    val stats = Stats[Double](INPUT)
    stats.laplaceMean(1) should be (1.11 +- 0.01)
    stats.laplaceMean(5) should be (0.77 +- 0.01)
  }

  it should s"$name Lidstone smoothing" in {
    show("$name Lidstone smoothing")
    val stats = Stats[Double](INPUT)
    stats.lidstoneMean(0.5, 1) should be (1.11 +- 0.01)
    stats.lidstoneMean(0.25, 1) should be (1.12 +- 0.01)
    stats.lidstoneMean(0.5, 5) should be (0.90 +- 0.01)
  }

  it should s"$name z-score transform" in {
    show(s"$name z-score transform")
    val stats = Stats[Double](INPUT)

    val zScoreResults = stats.zScore
    zScoreResults.head should be (-0.150 +- 0.02)
    zScoreResults(2) should be (-1.347 +- 0.02)
  }
}


// -------------------------------  EOF -------------------------------------------
