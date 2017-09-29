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

import scala.util.Try


/**
  * Unit test for the E-greedy algorithm
  * @author Patrick Nicolas
  * @version 0.99.2
  */
final class EGreedyTest extends FlatSpec with Matchers with Logging {
  protected val name: String = "Epsilon greedy"
  private var last_cumulative_regret_03 = -1.0
  private var last_cumulative_regret_06 = -1.0
  private var last_cumulative_regret_09 = -1.0

  it should s"$name evaluation with epsilon 0.3 and 10 arms" in {
    show("$name evaluation with epsilon 0.3 and 10 arms")

    val epsilon = 0.3
    val numArms = 10
    val numActions = 30

    val profile = getProfile(epsilon, numArms, numActions)
    show(s"$name ${profile.mkString(", ")}")
    last_cumulative_regret_03 = profile.last

    last_cumulative_regret_03 > profile(10) should be (true)
    show(s"$name last cumulative regret value: $last_cumulative_regret_03")
  }

  it should s"$name evaluation with epsilon 0.6 and 10 arms" in {
    show(s"$name evaluation with epsilon 0.6 and 10 arms")

    val epsilon = 0.6
    val numArms = 10
    val numActions = 30

    val profile = getProfile(epsilon, numArms, numActions)
    show(s"$name ${profile.mkString(", ")}")
    last_cumulative_regret_06 = profile.last

    last_cumulative_regret_06 > profile(10) should be (true)
    show(s"$name last cumulative regret value: $last_cumulative_regret_06")
    last_cumulative_regret_06 > last_cumulative_regret_03 should be (true)
  }

  it should s"$name evaluation with epsilon 0.9 and 10 arms" in {
    show(s"$name evaluation with epsilon 0.9 and 10 arms")

    val epsilon = 0.9
    val numArms = 10
    val numActions = 30

    val profile = getProfile(epsilon, numArms, numActions)
    show(s"$name ${profile.mkString(", ")}")
    last_cumulative_regret_09 = profile.last

    last_cumulative_regret_09 > profile(10) should be (true)
    show(s"$name last cumulative regret value: $last_cumulative_regret_09")
    last_cumulative_regret_09 > last_cumulative_regret_06 should be (true)
  }


  private def getProfile(epsilon: Double, numArms: Int, numActions: Int): IndexedSeq[Double] = {
    val arms = List.tabulate(numArms)(n => new Arm { override val id: Int = n } )
    val epsilonGreedy = new EGreedy[Arm](epsilon, arms)

    val pfnCumulRegret = epsilonGreedy.|>

    (0 until numActions).map( n => {
      if(n < 5)
       pfnCumulRegret(arms(3)).getOrElse(-1.0)
      else
        pfnCumulRegret(arms(5)).getOrElse(-1.0)
    })
  }
}


// ---------------------------------  EOF --------------------------------------------
