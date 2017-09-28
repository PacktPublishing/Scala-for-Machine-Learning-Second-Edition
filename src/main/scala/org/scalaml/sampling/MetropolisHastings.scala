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

import MetropolisHastings._

import scala.collection.mutable.ArrayBuffer


/**
  * Tracing class to collect statistics during execution of the Metropolis-Hastings algorithm.
  * @author Patrick Nicolas
  * @version 0.99.2
  */
private[scalaml] class Trace {
  private[this] val profile = ArrayBuffer[State]()
  private[this] var accepted: Double = -1.0

  def +=(s: State): Unit = profile.append(s)

  def +=(): Unit = accepted += 1

  final def acceptedRate(iters: Int): Double = accepted.toDouble / iters

  override def toString: String = s"History: ${profile.map(_.mkString(", ")).mkString("\n")}\nAccepted: $accepted"
}


/**
  * Implementation of a the generic Metropolis Hastings algorithm for a random walk applied to a state
  * @author Patrick Nicolas
  * @version 0.99.2
  * @param p The target probability distribution
  * @param q The proposal distribution that encapsulates the joint probability of transition between two states
  * @param proposer  The state transistion
  * @param rand Uniform random generator
  * @see Scala for Machine Learning Chapter 8, Monte Carlo Inference - Markov Chain Monte Carlo
  */
private[scalaml] class MetropolisHastings(
  p: State => Double,
  q: (State, State) => Double,
  proposer: State => State,
  rand: () => Double
) {
  import Math._


  /**
    * Execution of the Markov chain Monte Carlo using Metropolis-Hastings algorithm, for a single
    * variable continuous probability distribution.
    * @param initial initial value
    * @param numIters Number of iteration or steps used during the Random walk
    * @return trace or record of the random walk.
    */
  def mcmc(initial: Double, numIters: Int): Trace = mcmc(Vector[Double](initial), numIters)

  /**
    * Execution of the Markov chain Monte Carlo using Metropolis-Hastings algorithm. This execution
    * use a generic State as a value (multi-variate feature).
    * @param initial Initial state
    * @param numIters Number of iteration or steps used during the Random walk
    * @return trace or record of the random walk.
    */
  def mcmc(initial: State, numIters: Int): Trace = {
    var s: State = initial
    var ps = p(initial)
    val trace = new Trace

    (0 until numIters).foreach(iter => {
      val sPrime = proposer(s)
      val psPrime = p(sPrime)
        // Applies the computation of the acceptance criteria
      val logAlpha = psPrime - ps + q(sPrime, s) - q(s, sPrime)

      if (logAlpha >= 0 || rand() < exp(logAlpha)) { // Proposal q accepted
        trace.+=()
        s = sPrime
        ps = psPrime
      }
      trace.+=(s)
    })

    trace
  }
}



/**
  * Implementation of the Metropolis Hastings algorithm for uni-variate continuous probability distribution
  * @author Patrick Nicolas
  * @version 0.99.2
  *
  * @param p The target probability distribution
  * @param q The proposal distribution that encapsulates the joint probability of transition between two states
  * @param proposer  The state transistion
  * @param rand Uniform random generator
  * @see Scala for Machine Learning Chapter 8, Monte Carlo Inference - Markov Chain Monte Carlo
  */
private[scalaml] final class OneMetropolisHastings(
  p: Double => Double,
  q: (Double, Double) => Double,
  proposer: Double => Double,
  rand: () => Double
) extends MetropolisHastings((s: State) => p(s.head),
  (s: State, sPrime: State) => q(s.head, sPrime.head),
  (s: State) => Vector[Double](proposer(s.head)),
  rand)


private[scalaml] object MetropolisHastings {
  type State = Vector[Double]
}
