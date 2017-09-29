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


/**
  * Define the Sparsity characteristics
  * @note lambda is the sparsity penalization factor and beta is the secondary learning rate
  * @version 0.99.2
  * @author Patrick Nicolas
  * @see Scala for Machine Learning Chapter 11 Deep learning/ Autoencoder
  */
private[scalaml] trait Sparsity {
  protected[this] val lambda: Double
  protected[this] val beta: Double
  protected[this] val rhoLayer: Array[Double]
  protected[this] var rhoLayerPrime: Array[Double] = rhoLayer
  private val lambda_1 = 1.0 - lambda

  @throws(classOf[IllegalArgumentException])
  protected def update(activeValues: Array[Double]): Unit = {
    require(activeValues.size > 0, "Cannot initialize sparsity with undefined values")

    rhoLayerPrime = activeValues.indices.map(n => lambda * rhoLayer(n) + lambda_1 * activeValues(n)).toArray
  }

  protected def penalize(bias: Double, index: Int): Double = bias - beta*(rhoLayer(index)- rhoLayerPrime(index))
}


// --------------------------------  EOF --------------------------------------------------
