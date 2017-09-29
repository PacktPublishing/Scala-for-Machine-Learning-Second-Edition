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

import breeze.stats.distributions.Beta


/**
  * Generic Arm used in several exploration/exploitation algorithm.
  * @author Patrick Nicolas
  * @version 0.99.2
  */
private[scalaml] trait Arm {
  val id: Int
  var successes: Int = 1
  var failures: Int = 1

  def mean: Double = successes.toDouble/(successes + failures)

  /**
    * Update the Bernoulli distribution (successes/failures) after this arm
    * been played and received a reward
    */
  def apply(winner: Arm): Unit =
    if (id == winner.id) {
      winner.successes += 1
      failures += 1
    }
    else {
      winner.failures += 1
      successes += 1
    }

  override def toString: String = s"id=$id, count${successes + failures}, mean=$mean"
}

/**
  *
  */
private[scalaml] trait BetaArm extends Arm {
  private[this] var pdf = new Beta(1, 1)

  /**
    * Update the Bernoulli distribution (successes/failures) after this arm
    * been played and received a reward
    */
  override def apply(winner: Arm): Unit = {
    super.apply(winner)
    pdf = new Beta(successes, failures)
  }

  @inline
  final def sample: Double = pdf.draw
}


/**
  *
  */
private[scalaml] trait CountedArm extends Arm {
  import Math._
  private[this] var cnt: Int = _

  def select: Unit = cnt += 1

  final def score(numActions: Int): Double = mean + sqrt(2.0*log(numActions)/cnt)
}

// ------------------------ EOF -----------------------------------------------------------

