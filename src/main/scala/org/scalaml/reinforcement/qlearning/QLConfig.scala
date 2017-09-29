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
package org.scalaml.reinforcement.qlearning

import org.scalaml.core.Design.Config

/**
 * Parameterized class that defines the configuration parameters for the Q-Learning.
 * @constructor Create a configuration for the Q-learning algorithm.
 * @throws IllegalArgumentException if ''alpha'', ''gamma'', ''episodeLength'', ''numberEpisode''
 * or ''minCoverage'' is out of range.
 * @param alpha Learning rate for the Q-Learning algorithm.
 * @param gamma  Discount rate for the Q-Learning algorithm.
 * @param episodeLength Maximum number of states visited per episode.
 * @param numEpisodes  Number of episodes used during training.
 * @param minCoverage Minimum coverage allowed during the training of the Q-learning model.
 * The coverage is the percentage of episodes for which the goal state is reached.
 *
 * @author Patrick Nicolas
 * @since 0.98.1 January 19, 2014
 * @version 0.99.2
 * @see Scala for Machine Learning Chapter 11 ''Reinforcement learning'' / Q-learning
 */
@throws(classOf[IllegalArgumentException])
private[scalaml] case class QLConfig(
    alpha: Double,
    gamma: Double,
    episodeLength: Int,
    numEpisodes: Int,
    minCoverage: Double
) extends Config {
  import QLConfig._

  check(alpha, gamma, episodeLength, numEpisodes, minCoverage)
}

/**
 * Companion object for the configuration of the Q-learning algorithm. This singleton defines
 * the constructor for the QLConfig class and validates its parameters.
 * @author Patrick Nicolas
 * @since 0.98 January 19, 2014
 * @version 0.99.2
 * @see Scala for Machine Learning Chapter 15 Reinforcement learning/Q-learning
 */
private[scalaml] object QLConfig {
  private val NO_MIN_COVERAGE = 0.0

  private val MAX_EPISODES = 1000

  private def check(
    alpha: Double,
    gamma: Double,
    episodeLength: Int,
    numEpisodes: Int,
    minCoverage: Double
  ): Unit = {

    require(
      alpha > 0.0 && alpha < 1.0,
      s"QLConfig found alpha: $alpha required > 0.0 and < 1.0"
    )
    require(
      gamma > 0.0 && gamma < 1.0,
      s"QLConfig found gamma $gamma  required > 0.0 and < 1.0"
    )
    require(
      numEpisodes > 2 && numEpisodes < MAX_EPISODES,
      s"QLConfig found $numEpisodes $numEpisodes required > 2 and < $MAX_EPISODES"
    )
    require(
      minCoverage >= 0.0 && minCoverage <= 1.0,
      s"QLConfig found $minCoverage $minCoverage required > 0 and <= 1.0"
    )
  }
}

// ----------------------------  EOF --------------------------------------------------------------
