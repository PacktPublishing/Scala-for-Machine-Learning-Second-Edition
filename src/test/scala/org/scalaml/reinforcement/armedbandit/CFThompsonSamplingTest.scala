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
package org.scalaml.reinforcement.armedbandit

import org.scalaml.Logging
import org.scalatest.{FlatSpec, Matchers}


/**
  * Unit test for the Thompson Sampling
  */
final class CFThompsonSamplingTest extends FlatSpec with Matchers with Logging {
  protected val name: String = "Context-free Thompson sampling"
  private var last_cumulative_regret_002 = -1.0
  private var last_cumulative_regret_005 = -1.0
  private var last_cumulative_regret_01 = -1.0

  final private val numArms = 20
  final private val numActions = 180

  it should s"$name evaluation epsilon 0.02, 20 arms, 180 actions" in {
    show("$name evaluation epsilon 0.02, 20 arms, 180 actions")

    val epsilon = 0.02

    val profile = evalCFThompsonSampling(epsilon)
    show(s"$name profile ${profile.mkString(", ")}")
    last_cumulative_regret_002 = profile.last
    profile.last > profile(10) should be (true)
  }

  it should s"$name evaluation epsilon 0.05, 20 arms, 180 actions" in {
    show("$name evaluation epsilon 0.05, 20 arms, 180 actions")
    val epsilon = 0.05

    val profile = evalCFThompsonSampling(epsilon)
    show(s"$name profile ${profile.mkString(", ")}")
    last_cumulative_regret_005 = profile.last
    profile.last > profile(10) should be (true)

    show(s"$name last cumulative regret $last_cumulative_regret_005")
    last_cumulative_regret_005 > last_cumulative_regret_002 should be (true)
  }


  it should s"$name evaluation epsilon 0.1, 20 arms, 180 actions" in {
    show(s"$name evaluation epsilon 0.1, 20 arms, 180 actions")

    val epsilon = 0.1

    val profile = evalCFThompsonSampling(epsilon)
    show(s"$name profile ${profile.mkString(", ")}")
    last_cumulative_regret_01 = profile.last

    last_cumulative_regret_01 > last_cumulative_regret_005 should be (true)
    profile.last > profile(10) should be (true)
  }

  private def evalCFThompsonSampling(epsilon: Double): IndexedSeq[Double] = {
    val startPureExploration = (numArms<<1)
    val arms = List.tabulate(numArms)(n => new BetaArm { override val id: Int = n } )

    val cfThompsonSampling = new CFThompsonSampling[BetaArm](epsilon, arms)

    val pfnCFTS = cfThompsonSampling.|>
    (0 until numActions).map( n =>
      if(n >= startPureExploration)
        pfnCFTS(arms(6)).getOrElse(-1.0)
      else
        pfnCFTS(arms(n % numArms)).getOrElse(-1.0)
    )
  }
}


// ------------------------------  EOF --------------------------------
