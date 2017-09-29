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

import org.scalaml.core.ITransform
import scala.util.Try


/**
  * Implementation of the Upper Confidence Bounds algorithm
  *
  * @author Patrick Nicolas
  * @version 0.99.2
  * @param arms List of bandit arms
  * @tparam U Type of the arm
  */
private[scalaml] class UCB1[U <: CountedArm](override val arms: List[U]) extends ArmedBandit[U] {
  private[this] var numActions: Int = _

  override def select: U = {
    numActions += 1
    arms.sortBy(_.score(numActions)).head
  }

  /**
    * Update the number of successes and failures for each arms
    * @param successArm Arm that has been selected and actioned upon
    */
  override def apply(successArm: U): Unit = arms.foreach( _.apply(successArm) )
}

// -------------------------  EOF ------------------------------
