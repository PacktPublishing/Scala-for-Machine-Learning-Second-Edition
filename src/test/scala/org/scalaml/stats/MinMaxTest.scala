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

final class MinMaxTest extends FlatSpec with Matchers with Logging with Assertable {

  protected[this] val name = "Minimum-Maximum"

  val VALUES: Vector[Array[Double]] = Vector[Array[Double]](
    Array[Double](2.6, 1.7, 9.9),
    Array[Double](-2.9, 11.7, 29.9),
    Array[Double](0.6, -17.5, 50.5),
    Array[Double](12.0, 0.2, -34.8)
  )

  it should s"$name simple min-max values" in {
    show("$name simple min-max values")
    val mMxVector = new MinMaxVector(VALUES)
    val minMaxValues = mMxVector.minMaxVector.map(mnx => (mnx.min, mnx.max))

    val (min, max): (Double, Double) = minMaxValues.head
    min should be (-2.9)
    max should be (12.0)
    show(s"(min, max): ${minMaxValues.head}")

    val (min1, max1): (Double, Double) = minMaxValues(1)
    min1 should be (-17.5)
    max1 should be (11.7)
    show(s"min maxes ${minMaxValues.mkString(" - ")}")
  }

  it should s"$name normalize" in {
    show(s"$name normalize")
    val mMxVector = new MinMaxVector(VALUES)
    val minMaxValues = mMxVector.minMaxVector.map(mnx => (mnx.min, mnx.max))
    val normalizedMinMaxValue = mMxVector.normalize(0.0, 1.0)

    val (min, max): (Double, Double) = normalizedMinMaxValue.map(mnx => (mnx.min, mnx.max)).head
    show(s"min: $min max: $max")
    min should be (0.370 +- 0.02)
    max should be (0.656 +- 0.02)
  }
}
