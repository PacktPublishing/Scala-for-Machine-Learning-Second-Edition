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
package org.scalaml.ga.state

import org.scalaml.Predef.Context.ToDouble

import scala.util.Try
import org.scalaml.util.LoggingUtils._
import org.scalaml.ga.{Gene, Population}

/**
 * Trait that specializes the generic monitoring capabilities of ''Monitor''
 * The main function of the monitor is to initialize and monitor the
 * state of the genetic algorithm. Monitoring is only available to the
 * ''GASolver.|>'' predictive method
 * @tparam T type of the gene used in the genetic algorithm, with ''Gene'' as the
 * type upper-bound
 * @author Patrick Nicolas
 * @since 0.97 (Oct. 3, 2013)
 * @version 0.99.2
 * @see Scala for Machine learning Chap 13 ''Evolutionary Computing'' / Implementation
 *  / Solver
 */
private[scalaml] trait GAMonitor[U, T <: Gene[U]] extends Monitor[Double] {
  self: {
    def |> : PartialFunction[Population[U, T], Try[Population[U, T]]]
  } =>

  /**
   * Function to monitor the state and attributes of the population of
   * chromosomes during the execution of the genetic algorithm
   */
  protected val monitor: Option[Population[U, T] => Unit]
  private[this] var state: GAState = GA_NOT_RUNNING

  /**
   * Test of the genetic algorithm is ready to run
   * @return true if GA is not running, false otherwise
   */
  @inline
  final def isReady: Boolean = state.equals(GA_NOT_RUNNING)

  /**
   * Initialize the state of the genetic algorithm as running
   */
  def start(): Unit = state = GA_RUNNING

  /**
   * Method that implements the exit condition for the execution of the
   * genetic algorithm. The method is responsible for implementing the
   * convergence criteria
   * @param population The current population of chromosomes
   * @param remainingCycles The number of cycles left in the execution of the
   * genetic algorithm
   * @return true if genetic algorithm is still running, false otherwise
   */
  def isComplete(population: Population[U, T], remainingCycles: Int): Boolean = {
    state = if (population.isEmpty)
      GA_FAILED(s"GASolver.converge failed at $remainingCycles cycle")
    else if (remainingCycles < 1)
      GA_NO_CONVERGENCE(s"GASolver.converge failed")
    else {
      monitor.foreach(_(population))
      GA_RUNNING
    }
    !(state == GA_RUNNING)
  }
}

// ---------------------------------  EOF ----------------------