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

import AE._
import org.scalaml.stats.TSeries.zipWithShift1
import org.scalaml.supervised.mlp.{MLPConnection, MLPModel, MLPNetwork}


/**
  * @author Patrick Nicolas
  * @param config Configuration parameters for the AE model
  * @param topology Topology {input, hidden and output layer} for the AE model
  * @param model if the AE has already been created
  * @version 0.99.2
  * @see Scala for Machine Learning Chapter 11 Deep learning / Auto-encoder
  * @see org.scalaml.supervised.nnet.mlp.MLPNetwork
  */
private[scalaml] final class AENetwork protected (
  config: AEConfig,
  topology: Array[Int],
  model: Option[MLPModel] = None) extends MLPNetwork(config, topology, model) {

  /*
 * Create a array of connection between layer. A connection is
 * made of multiple synapses.
 */
  override protected[this] val connections: Array[MLPConnection] = zipWithShift1(layers.toArray).map {
    case (src, dst) => AEConnection(config, src, dst, model)
  }
}

/**
  *
  */
private[scalaml] object AENetwork  {
  def apply(
    config: AEConfig,
    topology: Array[Int],
    model: Option[MLPModel] = None): AENetwork =
    new AENetwork(config, topology, model)
}

// --------------------------------------   EOF ----------------------------------------------------------------------
