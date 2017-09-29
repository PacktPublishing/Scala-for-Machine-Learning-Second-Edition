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
package org.scalaml.ga

import org.apache.log4j.Logger
import org.scalaml.Predef.Context.ToDouble
import org.scalaml.core.ETransform
import org.scalaml.ga.state._

import scala.util.Try

/**
 * Class to select the best solution or Chromosome from an initial population
 * or genes pool using a set of policies for selection, mutation and crossover of
 * chomosomes. The client code initialize the GA solver with either an initialized
 * population or a function () => Population{T] that initialize the population. THe
 * class has only one public method search.
 * @tparam T Type of the gene (inherited from '''Gene''')
 * @constructor Create a generic GA-based solver.
 * @param config  Configuration parameters for the GA algorithm. The configuration is used as
 * a model for the explicit data transformation.
 * @param score Scoring method for the chromosomes of this population
 * @param _monitor optional method to monitor the state and size of the population during
 * reproduction.
 *
 * @author Patrick Nicolas
 * @since 0.97.1 August 29, 2013
 * @version 0.99.2
 * @see Scala for Machine Learning Chapter 10 Genetic Algorithm
 * @see http://www.kddresearch.org/Publications/Book-Chapters/Hs5.pdf
 */
final private[scalaml] class GASolver[U: ToDouble, T <: Gene[U]] protected (
    config: GAConfig,
    score: Chromosome[U, T] => Unit,
    tracker: Option[Population[U, T] => Unit]
) extends ETransform[Population[U, T], Population[U, T]](config) with GAMonitor[U, T] {

  protected val monitor: Option[Population[U, T] => Unit] = tracker
  protected val logger = Logger.getLogger("GASolver")

  /**
   * Method to resolve any optimization problem using a function to generate
   * a population of Chromosomes, instead an existing initialized population
   * @param initialize Function to generate the chromosomes of a population
   * @throws IllegalArgumentException If the initialization or chromosome generation
   * function is undefined
   */
  def |>(initialize: () => Population[U, T]): Try[Population[U, T]] = this.|>(initialize())

  /**
   * This method leverages the genetic algorithm reproduction cycle to select the fittest
   * chromosomes (or solutions candidate) after a predefined number of reproduction cycles
   *
   * Convergence criteria used to end the reproduction cycle is somewhat dependent of the
   * problem or domain. This implementation makes sense for the exercise in the book
   * chapter 10. It needs to be modified to a specific application. This implementation relies on
   * the tail recursion
   * @throws MatchError if the population is empty
   * @return PartialFunction with a parameterized population as input and the population
   * containing the fittest chromosomes as output.
   */
  override def |> : PartialFunction[Population[U, T], Try[Population[U, T]]] = {
    case population: Population[U, T] if population.size > 1 && isReady =>

      start()
      // Create a reproduction cycle manager with a scoring function
      val reproduction = Reproduction[U, T](score)

      @scala.annotation.tailrec
      def reproduce(population: Population[U, T], n: Int = 0): Population[U, T] =
        if (!reproduction.mate(population, config, n))
          population
        else if (isComplete(population, config.maxCycles - n))
          population
        else
          reproduce(population, n + 1)

      reproduce(population)

      // The population is returned no matter what..
      population.select(score, 1.0)
      Try(population)
  }
}

/**
 * Object companion for the Solve that defines the two constructors
 * @author Patrick Nicolas
 * @since August 29, 2013
 * @note Scala for Machine Learning Chapter 10 Genetic Algorithm
 */
private[scalaml] object GASolver {
  final val GA_COUNTER = "AvCost"

  /**
   * Default constructor for the Genetic Algorithm (class GASolver)
   * @param config  Configuration parameters for the GA algorithm
   * @param score Scoring method for the chromosomes of this population
   * @param monitor optional monitoring function that display the content or the various
   * metric associated to the current population
   */
  def apply[U: ToDouble, T <: Gene[U]](
    config: GAConfig,
    score: Chromosome[U, T] => Unit,
    monitor: Option[Population[U, T] => Unit]
  ): GASolver[U, T] = new GASolver[U, T](config, score, monitor)

  /**
   * Default constructor for the Genetic Algorithm (class GASolver)
   * @param config  Configuration parameters for the GA algorithm
   * @param score Scoring method for the chromosomes of this population
   */
  def apply[U: ToDouble, T <: Gene[U]](
    config: GAConfig,
    score: Chromosome[U, T] => Unit
  ): GASolver[U, T] = new GASolver[U, T](config, score, None)

  /**
   * Constructor for the Genetic Algorithm (class GASolver) with undefined scoring function
   * @param config  Configuration parameters for the GA algorithm
   */
  def apply[U: ToDouble, T <: Gene[U]](config: GAConfig): GASolver[U, T] =
    new GASolver[U, T](config, (c: Chromosome[U, T]) => Unit, None)
}

// ---------------------------  EOF -----------------------------------------------