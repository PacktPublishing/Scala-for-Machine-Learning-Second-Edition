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

import scala.util.Random

import org.scalaml.stats.TSeries._
import org.scalaml.util.FormatUtils
import MLPModel._, FormatUtils._, Math._

/**
 * Class that defines the connection between two sequential layers in a Multi-layer Perceptron.
 * A layer can be
 * - Input values
 * - Hidden
 * - Output values
 *
 * The connections are composed of all the synapses between
 * any neuron or variable of each layer.The Synapse is defined as a nested tuple(Double, Double)
 * tuple (weights, delta_Weights)
 * @constructor Create a MLP connection between two consecutive neural layer.
 * @param config  Configuration for the Multi-layer Perceptron.
 * @param src  Source (or input or upstream) neural layer to this connection
 * @param dst  Destination (or output or downstream) neural layer for this connection.
 * @param  mode Operating mode or objective of the Neural Network (binary classification,
 * regression...)
 *
 * @author Patrick Nicolas
 * @since 0.98.1 May 5, 2014
 * @version 0.99.2
 * @see Scala for Machine Learning Chapter 10 Multilayer perceptron / Model definition
 */
private[scalaml] class MLPConnection protected (
    config: MLPConfig,
    src: MLPLayer,
    dst: MLPLayer,
    model: Option[MLPModel]
)(implicit mode: MLP.MLPMode) {
  import MLPConnection._

  /*
		 * Initialize the matrix (Array of Array) of Synapse by generating
		 * a random value between 0 and ''boundary''. The boundary is computed as the inverse
		 * of the square root of the number of output values + 1 (bias element).
		 *
		 * @note The constant BETA may have to be changed according to the type of data.
		 */
  protected[this] var synapses: MLPConnSynapses =
    if (!model.isDefined) {
      val boundary = BETA / sqrt(src.output.length + 1.0)
      Array.fill(dst.numNonBias)(Array.fill(src.numNodes)((Random.nextDouble * boundary, 0.00)))
    } else
      model.get.synapses(src.id)

  /**
   * Implement the forward propagation of input value. The output
   * value depends on the conversion selected for the output. If the output or destination
   * layer is a hidden layer, then the activation function is applied to the dot product of
   * weights and values. If the destination is the output layer, the output value is just
   * the dot product weights and values.
   */
  def connectionForwardPropagation(): Unit = {
    // Iterates over all the synapses, compute the dot product of the output values
    // and the weights and applies the activation function
    val _output = synapses.map(x => dst.activation(innerDouble(src.output, x.map(_._1))))
    // Apply the objective function (SoftMax,...) to the output layer
    dst.setOutput(_output)
  }

  /**
   * Access the identifier for the source and destination layers
   * @return tuple (source layer id, destination layer id)
   */
  @inline
  final def getLayerIds: (Int, Int) = (src.id, dst.id)

  @inline
  final def getSynapses: MLPConnSynapses = synapses

  /**
   * Implement the back propagation of output error (target - output). The method uses
   * the derivative of the logistic function to compute the delta value for the output of
   * the source layer.
   * @param delta Delta error from the downstream connection (Connection with the destination
   * layer as a source).
   * @return A new delta error to be backpropagated to the upstream connection through the source
   * layer.
   */
  def connectionBackpropagation(delta: Delta): Delta = {
    val inSynapses = if (delta.synapses.length > 0) delta.synapses else synapses

    // Invoke the destination layer to compute the appropriate delta error
    val connectionDelta = dst.delta(delta.loss, src.output, inSynapses)

    // Traverses the synapses and update the weights and gradient of weights
    val oldSynapses = synapses.indices.map(j =>
       synapses(j).indices.map( i => {
        val ndw = config.eta * connectionDelta.delta(j)(i)
        val (w, dw) = synapses(j)(i)
        (w + ndw - config.alpha * dw, ndw)
      }).toArray
    ).toArray
    synapses = oldSynapses

    // Return the new delta error for the next (upstream) connection if any.
    Delta(connectionDelta, synapses)
  }

  /**
   * Textual representation of this connection. The description list the
   * values of each synapse as a pair (weight, delta weight)
   */
  override def toString: String = {
    val descr = (0 until dst.numNodes).map(i => {

      (0 until src.numNodes).map(j => {
        val wij: MLPSynapse = synapses(i)(j)
        val weights_str = format(wij._1, "", MEDIUM)
        val dWeights_str = format(wij._2, "", MEDIUM)
        s"$i,$j: ($weights_str, $dWeights_str)  "
      }).mkString("\n")
    }).mkString("\n")

    s"\nConnections weights from layer ${src.id} to layer ${dst.id}\n $descr"
  }
}

/**
 * Companion object for the connection of Multi-layer perceptron.
 * @author Patrick Nicolas
 * @since 0.98.1 May 5, 2014
 * @version 0.99.2
 * @see Scala for Machine Learning Chapter 10 Multilayer perceptron / Model definition
 */
private[scalaml] object MLPConnection {
  /**
   * Dumping factor for the random initialization of weightss
   */
  final val BETA = 0.2

  /**
   * Constructor for an ''MLPConnection''
   * @param config  Configuration for the Multi-layer Perceptron.
   * @param src  Source (or input or upstream) neural layer to this connection
   * @param dst  Destination (or output or downstream) neural layer for this connection.
   * @param  mode Operating mode or objective of the Neural Network (binary classification, regression...)
   */
  def apply(
    config: MLPConfig,
    src: MLPLayer,
    dst: MLPLayer,
    model: Option[MLPModel] = None
  )(implicit mode: MLP.MLPMode): MLPConnection =
    new MLPConnection(config, src, dst, model)

}

// -------------------------------------  EOF ------------------------------------------------