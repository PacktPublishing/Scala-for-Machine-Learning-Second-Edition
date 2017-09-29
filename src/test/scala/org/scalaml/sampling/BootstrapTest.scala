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

import org.apache.commons.math3.distribution.{NormalDistribution, RealDistribution}
import org.scalaml.Logging
import org.scalatest.{FlatSpec, Matchers}
import scala.collection.mutable.ArrayBuffer
import scala.util.Random


/**
  * Unit test for the Bootstrap sampling replicates
  */
final class BootstrapTest extends FlatSpec with Matchers with Logging {
  protected val name = "Bootstrap sampling replicates"
  final val NumReplicates1 = 256
  final val NumReplicates2 = 1024
  final val NumDataPoints = 10000

  private def bootstrapEvaluation(
    dist: RealDistribution,
    random: Random,
    coefs: (Double, Double),
    numReplicates: Int
  ): (Double, Double) = {

    val input = (0 until NumDataPoints)./:(new ArrayBuffer[(Double, Double)])(
      ( buf, _ ) => {
        val (a, b) = coefs
        val x = a * random.nextDouble - b
        buf += ( (x, dist.density(x)) )
      }
      ).toVector

      // Bootstrap for the statistisx
    val bootstrap = new Bootstrap(
      numReplicates,
      (x: Vector[Double]) => x.sum/x.length,
      input.map( _._2 ),
      (rLen: Int) => new Random( System.currentTimeMillis).nextInt(rLen)
    )
    (bootstrap.mean, bootstrap.error)
  }

  it should s"$name over a input with the distribution a*r + b $NumReplicates1 replicates" in {
    import Math._
    show(s"$name over a input with the distribution a*r + b $NumReplicates1 replicates")

    val (meanNormal, errorNormal) = bootstrapEvaluation(
      new NormalDistribution,
      new scala.util.Random,
      (5.0, 2.5),
      NumReplicates1
    )
    val expectedMean = 0.185
    show(s"$name meanNormal $meanNormal error $errorNormal")

    abs(expectedMean - meanNormal) < 0.05 should be (true)
    abs(errorNormal) < 0.05 should be (true)
  }

  it should s"$name over a input with the distribution a*r + b $NumReplicates2 replicates" in {
    import Math._
    show("$name over a input with the distribution a*r + b $NumReplicates2 replicates")

    val (meanNormal, errorNormal) = bootstrapEvaluation(
      new NormalDistribution,
      new scala.util.Random,
      (5.0, 2.5),
      NumReplicates2
    )
    val expectedMean = 0.185
    show(s"$name meanNormal $meanNormal error $errorNormal")

    abs(expectedMean - meanNormal) < 0.05 should be (true)
    abs(errorNormal) < 0.05 should be (true)
  }
}

// -----------------------------------  EOF -------------------------------------------