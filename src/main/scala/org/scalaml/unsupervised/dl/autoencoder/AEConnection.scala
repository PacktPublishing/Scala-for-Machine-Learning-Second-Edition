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
package org.scalaml.unsupervised.dl.autoencoder

import org.scalaml.supervised.mlp._
import AE._
import org.scalaml.stats.TSeries.innerDouble
import org.scalaml.supervised.mlp.{Delta, MLPConnection, MLPLayer, MLPModel}


/**
  * Definition of a connection for an Auto-encoder. The class reuses components of the Multi-layer perceptron
  * @author Patrick Nicolas
  * @param config  Configuration of the neural network
  * @param src  Source (or input or upstream) neural layer to this connection
  * @param dst  Destination (or output or downstream) neural layer for this connection.
  * @param model MLPmodel generates
  * @see Scala for Machine Learning Chapter 11 Deep learning / Auto-encoder
  * @see org.scalaml.supervised.nnet.mlp.MLPConnection
  */
private[scalaml] final class AEConnection protected (
  config: AEConfig,
  src: MLPLayer,
  dst: MLPLayer,
  model: Option[MLPModel])
  extends MLPConnection(config, src, dst, model) with Sparsity {

  protected[this] val lambda: Double = config.lambda
  protected[this] val beta: Double = config.beta
  protected[this] val rhoLayer: Array[Double] = Array.ofDim[Double](src.output.length)
  update(rhoLayer)

  /**
    * Implement the forward propagation of input value. The output
    * value depends on the conversion selected for the output. If the output or destination
    * layer is a hidden layer, then the activation function is applied to the dot product of
    * weights and values. If the destination is the output layer, the output value is just
    * the dot product weights and values.
    */
  override def connectionForwardPropagation(): Unit = {
    // Iterates over all the synapses, compute the dot product of the output values
    // and the weights and applies the activation function
    val _output = synapses.indices.map(n => {
      val weights = synapses(n).map(_._1)
      weights(0) = penalize(weights.head, n)
      dst.activation(innerDouble(src.output, weights))
    }).toArray

    update(_output)
    // Apply the objective function (SoftMax,...) to the output layer
    dst.setOutput(_output)
  }

  /**
    * Implement the back propagation of output error (target - output). The method uses
    * the derivative of the logistic function to compute the delta value for the output of
    * the source layer.
    * @param delta Delta error from the downstream connection (Connection with the destination
    * layer as a source).
    * @return A new delta error to be backpropagated to the upstream connection through the source
    * layer.
    */
  override def connectionBackpropagation(delta: Delta): Delta = {
    val inputSynapses = if (delta.synapses.length > 0) delta.synapses else synapses

    // Invoke the destination layer to compute the appropriate delta error
    val connectionDelta = dst.delta(delta.loss, src.output, inputSynapses)

    // Traverses the synapses and update the weights and gradient of weights
    synapses = synapses.indices.map(j =>
      synapses(j).indices.map( i => {
        val ndw = config.eta * connectionDelta.delta(j)(i)
        val (w, dw) = synapses(j)(i)
        (w + ndw - config.alpha * dw, ndw)
      }).toArray
    ).toArray

    // Return the new delta error for the next (upstream) connection if any.
    Delta(connectionDelta, synapses)
  }
}


private[scalaml] object AEConnection {
  /**
    * Constructor for an ''AEConnection''
    * @param config  Configuration for the Auto-encoder.
    * @param src  Source (or input or upstream) neural layer to this connection
    * @param dst  Destination (or output or downstream) neural layer for this connection.
    */
  def apply(
    config: AEConfig,
    src: MLPLayer,
    dst: MLPLayer,
    model: Option[MLPModel] = None): AEConnection = new AEConnection(config, src, dst, model)
}




// ---------------------------------   EOF --------------------------------------------------


