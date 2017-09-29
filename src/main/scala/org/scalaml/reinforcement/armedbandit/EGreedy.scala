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

import scala.util.{Random, Try}


/**
  *
  * @param epsilon
  * @param arms
  * @tparam U
  */
private[scalaml] class EGreedy[U <: Arm] (
  epsilon: Double,
  override val arms: List[U]) extends ArmedBandit[U] {
  import EGreedy._

  import Random._
  require(epsilon > MIN_EPSILON && epsilon < MAX_EPSILON, s"EGreedy $epsilon is out of bounds")
  require(arms.size > 0, "Number of arms is negative or null")

  //var cumulRegret: Double = _

  override def select: U = if(nextDouble < epsilon) arms(nextInt(arms.size)) else arms.sortBy(_.mean).head


  /**
    * Update the number of successes and failures for each arms
    * @param successArm Arm that has been selected and actioned upon
    */
  override def apply(successArm: U): Unit = arms.foreach( _.apply(successArm) )
}


private[scalaml] object EGreedy {
  final val MIN_EPSILON = 1e-10
  final val MAX_EPSILON = 1.0 - MIN_EPSILON
}

// ------------------------  EOF -------------------------------
