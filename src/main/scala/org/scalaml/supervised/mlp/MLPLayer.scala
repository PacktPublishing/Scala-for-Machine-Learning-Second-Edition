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

import scala.collection._

import org.scalaml.Predef._
import org.scalaml.util.FormatUtils._
import org.scalaml.stats.TSeries._
import MLPModel._

/**
 * Class that defines a input or hidden. A MLP layer is built using the
 * input vector and add an extra element (or neuron) to account for the intercept
 * weight w0.
 *
 * The MLP layer is fully defined by its rank in the Neuron Network with
 * input layer having id = 0 and the number of nodes (or neurons) and the activation
 * function to be used for its output values.
 *
 * @constructor Create an input or hidden layer for a multi-layer perceptron.
 * @throws IllegalArgumentException if the class parameters are incorrect
 * @param id Identifier or rank of the MLP layer in the network.
 * @param numNodes Number of nodes or neuron in the MLP layer.
 * @param activation Activation function for the neuron of this layer
 * @param mode Operating mode or objective for the MLP
 *
 * @author Patrick Nicolas
 * @since 0.98.1 May 6, 2014
 * @version 0.99.2
 * @see Scala for Machine Learning Chapter 10 Multilayer perceptron Model definition
 * @see org.scalaml.supervised.nnet.MLPOutLayer
 */
@throws(classOf[IllegalArgumentException])
private[scalaml] class MLPLayer(
    val id: Int,
    val numNodes: Int,
    val activation: Double => Double
)(implicit mode: MLP.MLPMode) {

  import MLPLayer._
  check(id, numNodes)

  /**
   * Array that contains the output value for this layer. The output values
   * are initialized to 1.0. All output except the bias (or first) neuron are
   * updated through the forward propagation of input values
   * @return array for output value for this layer
   */
  val output: Array[Double] = Array.fill(numNodes)(1.0)

  /**
   * Compute the range of indices output used in the back propagation. This
   * method is polymorphic
   * @return Range of indices for output values
   */
  def range: Range = Range(0, numNodes - 1)

  /**
   * Extract the number of non bias neurons for this layer
   * @return Number of non-bias neurons
   */
  def numNonBias: Int = numNodes - 1

  /**
   * Update the output values for the non-bias neurons
   * @param xt New values for the output of this layer
   * @throws IllegalArgumentException if the size of output values does not match the
   * number of non-bias neurons
   */
  @throws(classOf[IllegalArgumentException])
  def setOutput(xt: Array[Double]): Unit = {
    require(xt.length == numNodes - 1, s"MLPLayer found xt.size =${xt.size}, required ${numNodes -1}")
    xt.copyToArray(output, 1)
  }
  /**
   * Values of the output vector for this layer. It is used in
   * forward propagation.
   * @param x Input to the activation function
   * @return Activation value (tanh, sigmoid, ...)
   */
  def activate(x: Double): Double = activation(x)

  /**
   * Compute the delta error for the connectivity between hidden layers
   * and the input layer and hidden layer. The delta error is backpropagated from
   * the last hidden layer to the input layer.
   *
   * The delta error related to the connections between the last hidden layer and
   * the output layer is computed by ''MLPOutLayer.delta'' method. The weights of
   * the connection for which this layer is a source are extracted by transposing
   * the synapses matrix.
   * @param prevDelta delta error from the downstream connection (connection between this
   * layer as a source and the next layer
   * @param srcOut  Array of output values from the source layer associated with the
   * connection for which this layer is the destination
   * @param synapses of the connection for which this layer is a source
   * @return An instance of the delta error
   * @see org.scalaml.supervised.nnet.mlp.Delta
   */
  def delta(
    prevDelta: Array[Double],
    srcOut: Array[Double],
    synapses: MLPConnSynapses
  ): Delta = {
    import org.scalaml.util.MathUtils._
    val weights: Array[Array[Double]] = transpose(synapses.map(_.map(_._1))).tail

    val (loss, nDelta) = output.tail.zipWithIndex./:(mutable.ArrayBuffer[(Double, Array[Double])]()) {
      case (m, (zh, n)) =>
        val nextDelta = innerDouble(prevDelta, weights(n)) * zh * (1.0 - zh)
        m += ((nextDelta, srcOut.map(_ * nextDelta)) )
    }.toArray.unzip
    new Delta(loss, nDelta)
  }

  /**
   * Initialize the value of the input for this MLP layer.
   * @param x input vector for this layer.
   */
  def setInput(x: Array[Double]): Unit = x.copyToArray(output, output.length - x.length)

  /**
   * Test if this neural network layer is the output layer (last layer in the network).
   * @return true if this layer is the output layer, false, otherwise
   */
  @inline
  def isOutput: Boolean = false

  /**
   * Textual and formatted description of a layer in the Multi-layer perceptron
   */
  override def toString: String = {
    val desc = output.map(x => s"${format(x, "MLPLayer", SHORT)}").mkString(" ")
    s"\nLayer: $id output: $desc"
  }
}

/**
 * Companion object for the MLP layer used to define a default constructor
 * and validate its input parameters
 * @author Patrick Nicolas
 * @since 0.98.1 May 6, 2014
 * @version 0.99.2
 * @see Scala for Machine Learning Chapter 10 Multilayer perceptron / Model definition
 */
private[mlp] object MLPLayer {
  /**
   * Default constructor for MLPLayer
   * @param id Identifier or rank of the MLP layer in the network.
   * @param nNodes Number of elements or neuron in the MLP layer.
   * @param activation Activation function for the neuron of this layer
   * @param mode Operating mode or objective for the MLP
   * @throws IllegalArgumentException if the class parameters are incorrect
   * @return An instance of the input or hidden layer
   */
  def apply(
    id: Int,
    nNodes: Int,
    activation: Double => Double
  )(implicit mode: MLP.MLPMode): MLPLayer = new MLPLayer(id, nNodes, activation)

  def check(id: Int, len: Int): Unit = {
    require(id >= 0, s"MLPLayer Create a MLP layer with incorrect id: $id")
    require(len > 0, s"MLPLayer Create a MLP layer with incorrect length $len")
  }
}

/**
 * Class that defines the output layer. Contrary to the ''MLPLayer'' for input and hidden
 * layer, the output layer does not have a bias element.
 *
 * The MLP layer is fully defined by its rank in the Neuron Network with
 * input layer having id = 0 and the number of nodes (or neurons) and the activation
 * function to be used for its output values.
 *
 * The activation function for the neuron of the output layer is linear
 * @constructor Create an output layer for a multi-layer perceptron.
 * @throws IllegalArgumentException if the class parameters are incorrect
 * @param id Identifier or rank of the MLP layer in the network.
 * @param numNodes Number of nodes or neuron in the MLP layer.
 * @param mode Operating mode or objective for the MLP
 *
 * @author Patrick Nicolas
 * @since 0.98.1 May 6, 2014
 * @version 0.99.2
 * @see Scala for Machine Learning Chapter 10 Multilayer perceptron Model definition
 * @see org.scalaml.supervised.nnet.MLPLayer
 */
final protected class MLPOutLayer(
  id: Int,
  numNodes: Int
)(implicit mode: MLP.MLPMode) extends MLPLayer(id, numNodes, (x: Double) => x) {

  /**
   * Compute the range of indices output used in the back propagation. This
   * method is polymorphic
   * @return Range of indices for output values
   */
  override def range: Range = Range(0, numNodes)

  /**
   * Extract the number of non bias neurons for this layer
   * @return Number of non-bias neurons
   */
  override def numNonBias: Int = numNodes

  /**
   * Update the output values for the non-bias neurons
   * @param xt New values for the output of this layer
   * @throws IllegalArgumentException if the size of output values does not match the
   * number of non-bias neurons
   */
  override def setOutput(xt: Array[Double]): Unit = mode(xt).copyToArray(output)

  /**
   * Compute the delta error for the connection (synapses) between the last hidden layer
   * and the output layer. The delta error is backpropagated from the output layer to the
   * last hidden layer.
   *
   * The delta error related to the connections between hidden layers is computed by
   * ''MLPLayer.delta'' method.
   * @param error Array of errors = (expected value - predicted value) for each output
   * @param srcOut  Array of output values from the source layer associated with the
   * connection for which this layer is the destination
   * @param synapses of the connection for which this layer is a source
   * @return An instance of the delta error
   * @see org.scalaml.supervised.nnet.mlp.Delta
   */
  override def delta(error: Array[Double], srcOut: Array[Double], synapses: MLPConnSynapses): Delta = {
    val deltaValues = error./:(mutable.ArrayBuffer[Array[Double]]())((m, l) => m += (srcOut.map(_ * l)) )
    new Delta(error, deltaValues.toArray, synapses)
  }

  /**
   * Test if this neural network layer is the output layer (last layer in the network).
   * @return true if this layer is the output layer, false, otherwise
   */
  @inline
  override def isOutput: Boolean = true
}

/**
 * Companion object for the output layer of type ''MLPOutLayer''
 * @author Patrick Nicolas
 * @since 0.99 August 9 2015
 * @version 0.99.2
 * @see Scala for Machine Learning Chapter 10 Multilayer perceptron Model definition
 */
private[mlp] object MLPOutLayer {
  /**
   * Default constructor for ''MLPOutLayer''
   * @param id Identifier or rank of the MLP layer in the network.
   * @param nNodes Number of elements or neuron in the MLP layer.
   * @param mode Operating mode or objective for the MLP
   */
  def apply(
    id: Int,
    nNodes: Int
  )(implicit mode: MLP.MLPMode): MLPOutLayer = new MLPOutLayer(id, nNodes)
}

// -------------------------------------  EOF ------------------------------------------------