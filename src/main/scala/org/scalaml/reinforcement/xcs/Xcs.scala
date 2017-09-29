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
package org.scalaml.reinforcement.xcs

import scala.util.Try

import org.scalaml.ga._
import org.scalaml.trading.Signal
import org.scalaml.reinforcement.qlearning._
import org.scalaml.Predef._
import org.scalaml.core.ETransform

/**
 * Class that defines a sensor (or input stimuli) in an extended learning classifier system.
 * It is assumed that the XCS model monitors continuous values of type Double
 * @param id Identifier for the sensor or stimuli
 * @param value value of the stimuli or sensor.
 *
 * @author Patrick Nicolas
 * @since March 26, 2014
 * @note Scala for Machine Learning Chapter 11 Reinforcement learning / Extended learning
 * classifier systems
 */
case class XcsSensor(id: String, value: Double)

/**
 * Example of implementation of the XCS algorithm with a predefined
 * configuration and a set of training episode for the Q-Learning algorithm used to assign
 * credit to individual rules that improve the performance (or objective
 * function) of a system.
 * @constructor Create an extended learning classifiers system.
 * @throws IllegalArgumentException if the configuration, input information or training
 * episodes is undefined
 * @param config  Configuration for the XCS algorithm (GA and Q-Learning parameters)
 * @param population Initial population for the search space of classifiers
 * @param score	Chromosome scoring function
 * @param input Input for Q-learning state transition space QLSpace used in training
 * @author Patrick Nicolas
 * @since 0.98.1 March 26, 2014
 * @version 0.99.2
 * @note Scala for Machine Learning Chapter 15 Reinforcement learning / Extended learning
 * classifier systems
 */
final class Xcs(
    config: XcsConfig,
    population: Population[Double, Signal],
    score: Chromosome[Double, Signal] => Unit,
    input: Array[QLInput]
) extends ETransform[XcsSensor, List[XcsAction]](config) {

  import Xcs._

  check(population, input)

  val gaSolver = GASolver[Double, Signal](config.gaConfig, score)
  val features: Seq[Chromosome[Double, Signal]] = population.chromosomes.toSeq
  val qLearner = QLearning[Chromosome[Double, Signal]](config.qlConfig, extractGoals(input), input, features)

  private def extractGoals(input: Array[QLInput]): Int = -1
  private def computeNumStates(input: Array[QLInput]): Int = -1

  override def |> : PartialFunction[XcsSensor, Try[List[XcsAction]]] = {
    case _ => Try(List.empty[XcsAction])
  }
}

/**
 * Companion object for the extended learning classifier system.
 */
object Xcs {
  protected def check(
    population: Population[Double, Signal],
    input: Array[QLInput]
  ): Unit = {

    require(!input.isEmpty, "Xcs.check: Cannot create XCS with undefined state input")
    require(!population.isEmpty, "Xcs.check: Cannot create XCS with undefined population")
    require(
      population.size > 2,
      s"Xcs.check: Cannot create XCS with a population of size ${population.size}"
    )
  }
}

// ------------------------------------  EOF -----------------------------------------------------