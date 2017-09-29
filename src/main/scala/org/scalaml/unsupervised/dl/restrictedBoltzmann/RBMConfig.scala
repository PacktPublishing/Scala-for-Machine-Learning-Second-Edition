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
package org.scalaml.unsupervised.dl.restrictedBoltzmann


/**
  * Data class for the configuration of the binary restricted Boltzmann machine
  * @author Patrick Nicolas
  * @version 0.99.2
  *
  * @param learningRate learning rate
  * @param maxIters Maximum number of iterations allowed for the unsupervised training
  * @param tol tolerance or convergence criteria
  * @param penalty penalty function to compute the cost or penalty c = f(learningRate)
  * @see Scala for Machine Learning Chapter 11 Deep learning/ Restricted Boltzmann Machine
  */
@throws(classOf[IllegalArgumentException])
case class RBMConfig(
  learningRate: Double,
  maxIters: Int,
  tol: Double,
  penalty: Double => Double) {

  require(maxIters > 1, s"RBM configured with $maxIters maximum iterations, require >1")
  final def loss: Double = penalty(learningRate)
}

// ----------------------------  EOF -------------------------------------------
