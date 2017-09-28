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
package org.scalaml.supervised.mlp

import org.scalaml.Predef._
import MLPModel._

/**
 * Class that defines the Delta error used in the MLP back propagation
 * @constructor Create a Delta instance with an initial array of output errors,
 * the delta errors and the synapses the destination connection in the backpropagation and the
 * @param loss or error (expected value - predicted value) for each output variable
 * @param delta Matrix of delta errors computed in the previous connection by the
 * backpropagation algorithm
 * @param synapses Synapses (weights, delta weights) of the previous connection used in the
 * backpropagation.
 *
 * @author Patrick Nicolas
 * @since 0.99 August 9, 2014
 * @version 0.99.2
 * @see Scala for Machine Learning Chapter 10 Multi-layer perceptron Model definition
 * @see org.scalaml.supervised.nnet.MLPConnection
 */
private[scalaml] case class Delta(
    loss: Array[Double],
    delta: DblMatrix = Array.empty[Array[Double]],
    synapses: MLPConnSynapses = Array.empty[Array[MLPSynapse]]
) {

  override def toString: String = {
    val losses = loss.mkString(",")
    if (delta.length > 0) {
      val deltas = delta.map(_.mkString(", ")).mkString("\n")
      s"Losses: $losses\ndelta\n$deltas"
    } else s"Losses: $losses"
  }
}

private[scalaml] object Delta {
  def apply(connDelta: Delta, synapses: MLPConnSynapses): Delta = Delta(connDelta.loss, connDelta.delta, synapses)
}

// ----------------------- EOF ----------------------------

