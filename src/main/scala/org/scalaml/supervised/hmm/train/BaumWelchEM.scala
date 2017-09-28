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
package org.scalaml.supervised.hmm.train

import scala.util.Try
import scala.annotation.tailrec

import org.apache.log4j.Logger
import org.scalaml.supervised.hmm.{HMMConfig, HMMModel}
import org.scalaml.supervised.hmm.eval.{Alpha, Beta}
import org.scalaml.util.LoggingUtils
import LoggingUtils._

/**
 * Class that update the backward-forward lattice of observations and
 * hidden states for HMM using the Baum-Welch algorithm. The algorithm is used to
 * compute the likelihood of the conditional probability p(Y|X) during training. The
 * computation is performed as part of the instantiation of the class (type Option[Double] )
 *  @constructor Create a new Baum-Welch expectation maximization instance to train a model
 *  given a set of observations.
 *  @throws IllegalArgumentException if lambda, params and observations are undefined or
 *  eps is out of range
 *  @param config Configuration parameters class instance for the HMM
 *  @param obsSeq Observations defined as an array of Integer (or categorical data)
 *  @see org.scalaml.supervised.hmm.HMMModel
 *
 *  @author Patrick Nicolas
 *  @since 0.98 March 15, 2014
 *  @version 0.99.2
 *  @note Scala for Machine Learning Chapter 7 Sequential data models / Hidden Markov Model /
 *  Training
 */
@throws(classOf[IllegalArgumentException])
final private[scalaml] class BaumWelchEM protected (
    config: HMMConfig,
    obsSeq: Vector[Int]
) {
  import BaumWelchEM._

  check(config, obsSeq)
  protected val logger = Logger.getLogger("BaumWelchEM")

  /**
   * Initial state of the HMM model for training
   */
  val lambda = HMMModel(config)
  private[this] val gamma = new Gamma(lambda)
  private[this] val diGamma = new DiGamma(lambda)

  /**
   * Maximum likelihood (maximum log of the conditional probability) extracted from the training
   * @see Chapter 7 ''Sequential data models'' / Hidden Markov models / Training (CF-2)
   */
  val maxLikelihood: Option[Double] = {

    @tailrec
    def getLikelihood(likelihood: Double, index: Int): Double = {
      lambda.update(gamma, diGamma, obsSeq)

      frwrdBckwrdLattice match {
        case None => throw new IllegalStateException("BaumWelchHMM did not converge")
        case Some(estimate) =>
          val diff = likelihood - estimate
          if (diff < config.eps)
            estimate
          else if (index >= config.maxIters)
            throw new IllegalStateException("BaumWelchHMM did not converge")
          else
            getLikelihood(estimate, index + 1)
      }
    }

    val likelihood = frwrdBckwrdLattice.map(getLikelihood(_, 0))
    lambda.normalize()
    likelihood
  }


  /*
		 * Apply the alpha-beta passes to compute the probability Gamma of being in a
		 * given state for each observation [Formula M8], the probability DiGamma to being
		 * in a given state and transition to another given state for each observation [Formula M8]
		 * then update the state transition probabilities matrix A [Formula 10], the emission
		 * probabilities matrix B [Formula M11] and the initial probabilities PI [Formula M9]
		 */
  private def frwrdBckwrdLattice: Option[Double] = {
    // Compute the forward pass given the sequence of observations obs
    val pAlpha = Alpha(lambda, obsSeq)

    val beta = Beta(lambda, obsSeq).getTreillis
    // Compute the probabilities of a state given the

    gamma.update(pAlpha.getTreillis, beta)
    diGamma.update(pAlpha.getTreillis, beta, lambda.A, lambda.B, obsSeq)
    // Finally returns the likelihood
    pAlpha.alpha
  }
}

/**
 * Object companion for Baum_Welch algorithm that defines the constructor for BaumWelchEM
 * and validate its input parameters
 * @see org.scalaml.supervised.hmm.HMMModel
 * @author Patrick Nicolas
 * @since March 15, 2014
 * @note Scala for Machine Learning Chapter 7 Sequential data models / Hidden Markov Model /
 * Training
 */
private[scalaml] object BaumWelchEM {
  private val EPS = 1e-3

  /**
   * Default constructor for the BaumWelchEM class
   *  @param config Configuration parameters class instance for the HMM
   *  @param obsSeq Observations defined as an array of Integer (or categorical data)
   *  @return New instance of the Baum-Welch algorithm
   */
  def apply(config: HMMConfig, obsSeq: Vector[Int]): BaumWelchEM =
    new BaumWelchEM(config, obsSeq)

  private val EPS_LIMITS = (1e-8, 0.1)
  private val MAX_NUM_ITERS = 1024

  private def check(config: HMMConfig, obsSeq: Vector[Int]): Unit =
    require(obsSeq.length > 1, "BaumWelchEM obsSeq length found ${obsSeq.length} required > 1")

}
// -----------------------------  EOF --------------------------------