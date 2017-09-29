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
package org.scalaml.spark.extensibility

import java.lang.Math._

import org.scalaml.Logging
import org.scalatest.{FlatSpec, Matchers}
import org.apache.commons.math3.distribution.NormalDistribution
import org.scalaml.spark.{DatasetGenerator, SessionLifeCycle}

/**
  * Test suite to evalute Kullback-Leibler divergence between different Normal distribution
  */
final class KullbackLeiblerTest extends FlatSpec with Matchers with Logging {
  import DatasetGenerator._
  import SessionLifeCycle._
  protected[this] val name = "Spark/Kullback Leibler"

  implicit private val sessionLifeCycle = new SessionLifeCycle {}
  implicit private val sparkSession = sessionLifeCycle.sparkSession

  final val normalGenerator = new NormalDistribution
  final val NUM_DATA_POINTS = 5000

  final val normalData = toDSPairDouble(NUM_DATA_POINTS)((n: Int) => {
    val x = n.toDouble * 0.001
    (x, normalGenerator.density(x))
  })

  it should s"$name divergence using Normal distribution mu=0" in {
    show(s"$name divergence using Normal distribution mu=0")

    val mu = 0.0
    normalKL(mu) should be(0.0 +- 0.001)
  }

  it should s"$name divergence using Normal distribution mu=1.0" in {
    show(s"$name divergence using Normal distribution mu=1.0")

    val mu = 1.0
    normalKL(mu) should be(0.01 +- 0.01)
  }

  it should s"$name divergence using Normal distribution mu=2.0" in {
    show(s"$name divergence using Normal distribution mu=2.0")

    val mu = 2.0
    normalKL(mu) should be(-4.7 +- 0.2)
  }

  it should s"$name divergence using Normal distribution mu=3.0" in {
    show("$name divergence using Normal distribution mu=3.0")

    val mu = 3.0
    normalKL(mu) should be(-180.0 +- 2.0)
  }

  private def normalKL(mu: Double): Double = {
    import Math._

    val Inv2PI = 1.0 / sqrt(2.0 * PI)
    val pdf = (x: Double) => { val z = x - mu; Inv2PI * exp(-z * z) }

    val kullbackLeibler = KullbackLeibler(s"Normal mu=$mu", pdf)
    val klValue = kullbackLeibler.kl(normalData).head
    show(s"klValue for $mu $klValue")
    klValue
  }

  it should s"$name divergence using constant distribution" in {
    import Math._
    val kullbackLeibler = KullbackLeibler("Constant", (x: Double) => 2.0)
    val klValue = kullbackLeibler.kl(normalData).head
    klValue should be(-7028.0 +- 10.0)
  }

  it should s"$name formula" in {
    type DataSeq = Seq[(Double, Double)]
    val Eps = 1e-12
    val LogEps = log(Eps)
    def exec(xy: DataSeq, pdf: Double => Double): Double = {
      -xy./:(0.0) {
        case (s, (x, y)) => {
          val px = pdf(x)
          val z = if (abs(y) < Eps) px / Eps else px / y
          val t = if (z < Eps) LogEps else log(z)
          s + px * t
        }
      }
    }
    val h: Seq[(Double, Double)] = Seq.tabulate(1000)(
      (n: Int) => (n.toDouble * 0.001, normalGenerator.density(n.toDouble * 0.001))
    )
    val Inv2PI = 1.0 / sqrt(2.0 * PI)
    exec(h.iterator.toSeq, (x: Double) => Inv2PI * exp(-x * x)) should be(37.7 +- 0.1)
  }
}

// ---------------------------  EOF -----------------------------------------------