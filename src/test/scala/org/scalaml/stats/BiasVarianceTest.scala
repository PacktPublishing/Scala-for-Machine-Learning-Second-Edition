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
import org.scalaml.Predef.DblVec
import org.scalaml.util.Assertable
import org.scalaml.util.FormatUtils._
import org.scalatest.{FlatSpec, Matchers}

final class BiasVarianceTest extends FlatSpec with Matchers with Logging with Assertable {
  import scala.util.Random._, Math._

  protected[this] val name = "Bias-Variance decomposition"

  final val EXPECTED_BIAS = Array[Double](904.4, 401.9)
  final val EXPECTED_VARIANCE = Array[Double](1184.8, 789.9)

  it should s"$name evaluation bias and variance" in {
    show("$name evaluation bias and variance")

    // Any model that replicates the training set will overfit
    val training = (x: Double) => {
      val r1 = 0.45 * (nextDouble - 0.5)
      val r2 = 38.0 * (nextDouble - 0.5)
      0.2 * x * (1.0 + sin(x * 0.1 + r1)) + sin(x * 0.3) + r2

    }
    // Our target model used for emulating the validation data set
    val target = (x: Double) => 0.2 * x * (1.0 + sin(x * 0.1))

    // A list of models candidates including the overfitting model that
    /// match the training set.
    val models = List[(Double => Double, String)](
      ((x: Double) => 0.2 * x * (1.0 + 0.25 * sin(x * 0.1)), "Underfitting1"),
      ((x: Double) => 0.2 * x * (1.0 + 0.5 * sin(x * 0.1)), "Underfitting2"),
      ((x: Double) => training(x), "Overfitting"),
      ((x: Double) => 0.2 * x * (1.0 + sin(x * 0.1)), "Fitting")
    )

    val bv = BiasVariance(target, 160).fit(models.map(_._1))
    val (bias, variance) = bv.unzip

    bias.head should be (904.0 +- 1.0)
    bias(1) should be (402.0 +- 1.0)

    variance.head should be (1184.0 +- 2.0)
    variance(1) should be (790.0 +- 2.0)


    val result = format(bv.toVector, "Variance", "bias", SHORT)

    compareArray(bias.toArray.take(2), EXPECTED_BIAS, 1.0) should be (true)
    compareArray(variance.toArray.take(2), EXPECTED_VARIANCE, 1.0) should be (true)
  }
}
