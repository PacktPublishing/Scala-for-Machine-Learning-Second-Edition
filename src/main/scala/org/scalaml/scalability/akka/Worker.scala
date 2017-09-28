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

import org.apache.log4j.Logger
import akka.actor._

import org.scalaml.Predef._
import org.scalaml.scalability.akka.message._
import org.scalaml.util.{FormatUtils, LoggingUtils}
import FormatUtils._, Controller._, LoggingUtils._

/**
 * Worker actor responsible for transforming a time series using the
 * PipeOperator |>. The computation is initiated by the Master that acts
 * as the workflow controller.
 * @constructor Create a worker actor.
 * @param id Identifier or counter for the worker actors.
 * @param fct Data transformation function to be applied to a time series.
 *
 * @author Patrick Nicolas
 * @since 0.98.1 March 24, 2014
 * @see Scala for Machine Learning Chapter 16 Parallelism with Scala and Akka
 * @version 0.99.2
 */
final private[scalaml] class Worker(id: Int, fct: PfnTransform) extends Actor with Monitor[Double] {
  import Worker._
  check(id)

  protected val logger = Logger.getLogger("WorkerActor")
  override def preStart(): Unit = show(s"Worker${id}.preStart")
  override def postStop(): Unit = show(s"Worker${id}.postStop")

  /**
   * Event loop of the work actor that process two messages:
   * <ul>
   *  <li>Activate: to start processing this assigned partition</li>
   *  <li>Terminate: To stop this worker actor
   *  </ul>
   */
  override def receive = {
    case msg: Activate =>
      // Increment the messages id
      val msgId = msg.id + id
      show(s"Worker_${id}.receive:  Activate message $msgId")

      // Execute the data transformation
      val output: DblVec = fct(msg.xt).get
      show(results(output.take(NUM_DATAPOINTS_DISPLAY)))

      // Returns the results for processing this partition
      sender ! Completed(msgId, output)

    case _ => error(s"Worker${id}.receive Message not recognized")
  }

  private def results(output: DblVec): String =
    output.map(o => s"${format(0, "", MEDIUM)}").mkString(" ")
}

/**
 * Companion object for the worker actor. The singleton is used to validate
 * the parameters of the Worker class
 *
 * @author Patrick Nicolas
 * @since 0.98.1 March 24, 2014
 * @see Scala for Machine Learning Chapter 16 Parallelism with Scala and Akka
 * @version 0.99.2
 */
private[scalaml] object Worker {
  private def check(id: Int): Unit = require(id >= 0, s"Worker.check Id $id is out of range")
}

// ---------------------------------  EOF -------------------------