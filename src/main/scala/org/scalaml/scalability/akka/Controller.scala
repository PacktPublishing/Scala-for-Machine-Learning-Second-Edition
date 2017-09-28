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
package org.scalaml.scalability.akka

import scala.collection.mutable.ListBuffer
import scala.util.Try
import akka.actor.Actor
import org.scalaml.core.ETransform
import org.scalaml.Predef._
import Controller._
import org.scalaml.core.Design.Config

/**
 * Generic controller actor that defines the three key elements of a distributed
 * data transformation.
 *  @constructor Create a controller for data transformations:
 *  @throws IllegalArgumentException if the time series is undefined or empty
 *  @param xt Time series to be processed
 *  @param fct Data transformation of type PipeOperator
 *  @param nPartitions Number segments or partitions to be processed by workers.
 *
 *  @author Patrick Nicolas
 *  @since 0.98 March 30, 2014
 *  @see Scala for Machine Learning Chapter 16 Parallelism with Scala and Akka
 *  @version 0.99.2
 */
abstract private[scalaml] class Controller protected (
    val xt: DblVec,
    val fct: PfnTransform,
    val nPartitions: Int
) extends Actor {

  require(
    xt.nonEmpty,
    "Master.check Cannot create the master actor, undefined time series"
  )

  /**
   * Method to split a given time series into 'numPartitions' for concurrent processing
   * @return Sequence/Iterator of absolute index in the time series associated with each partition.
   * @throws IllegalArgumentException if the time series argument is undefined.
   */
  final def partition: Iterator[DblVec] = {
    // Compute the size of each partition
    val sz = (xt.size.toDouble / nPartitions).ceil.toInt
    xt.grouped(sz)
  }
}

private[scalaml] final class Aggregator(partitions: Int) {
  private val state = new ListBuffer[DblVec]

  def +=(x: DblVec): Boolean = {
    state.append(x)
    state.size == partitions
  }

  def clear(): Unit = state.clear()

  @inline
  final def completed: Boolean = state.size == partitions
}


/**
  *  @see Scala for Machine Learning Chapter 16 Parallelism with Scala and Akka
  *  @version 0.99.2
  */
private[scalaml] object Controller {
  final val MAX_NUM_DATAPOINTS = 256
  final val NUM_DATAPOINTS_DISPLAY = 12
  type PfnTransform = PartialFunction[DblVec, Try[DblVec]]
}

private[scalaml] object TransformTypes {
  abstract class DT[T <: Config](t: T) extends ETransform[DblVec, DblVec](t)
}

// -----------------------------------------  EOF ------------------------------------------------