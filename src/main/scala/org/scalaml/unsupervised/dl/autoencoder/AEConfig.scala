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

import org.scalaml.supervised.mlp.MLPConfig

/**
  * Configuration for the Sparse Autoencoder
  *
  * @author Patrick Nicolas
  * @param alpha  Momentum parameter used to adjust the value of the gradient of the weights
  * with previous value (smoothing)
  * @param eta   Learning rate ]0, 1] used in the computation of the gradient of the weights
  * during training
  * @param lambda Sparsity penalization term
  * @param beta Secondary learning scaling factor
  * @param numEpochs  Number of epochs or iterations allowed to train the weights/model
  * @param eps  Convergence criteria used as exit condition of the convergence toward optimum
  * weights that minimize the sum of squared error
  * @version 0.99.2
  * @see Scala for Machine Learning Chapter 11 Deep learning - Auto-encoder
  */
@throws(classOf[IllegalArgumentException])
private[scalaml] class AEConfig protected (
  alpha: Double,
  eta: Double,
  val lambda: Double,
  val beta: Double,
  numEpochs: Int,
  eps: Double) extends MLPConfig(alpha, eta, numEpochs, eps, Math.tanh ) {
    // Checking
  MLPConfig.check(alpha, eta, numEpochs)
  AEConfig.check(lambda, beta)
}

/**
  *
  */
private[scalaml] object AEConfig {
  final private val LAMBDA_LIMITS = (1e-10, 0.2)
  final private val BETA_LIMITS = (1e-5, 1.0)

  def apply(
    alpha: Double,
    eta: Double,
    lambda: Double,
    beta: Double,
    numEpochs: Int,
    eps: Double) = new AEConfig(alpha, eta, lambda, beta, numEpochs, eps)


  protected def check(lambda: Double, beta: Double): Unit = {
    require(
      lambda >= LAMBDA_LIMITS._1 && lambda <= LAMBDA_LIMITS._2,
      s"Sparsity penalty, lambda $lambda is out of bounds"
    )
    require(
      beta >= BETA_LIMITS._1 && beta <= BETA_LIMITS._2,
      s"Sparsity penalty, beta $beta is out of bounds"
    )
  }
}

// ----------------------------------------  EOF --------------------------------------------------------

