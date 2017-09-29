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
package org.scalaml.libraries.libsvm

import scala.collection.mutable.ArrayBuffer
import libsvm.{svm_problem, svm_node, svm, svm_model, svm_parameter}
import org.scalaml.supervised.svm.SVMModel

/**
  * Singleton adapter to LIBSVM library
  * @author Patrick Nicolas
  * @since 0.98 March 11, 2014
  * @version 0.99.2
  */
private[scalaml] object SVMAdapter {
  type SVMNodes = Array[Array[svm_node]]

  /**
    * Class that wraps the definition of a SVM problem as defined in LIBSVM
    * @param numObs Number of observations
    * @param expected Expected value for the feature vector
    */
  class SVMProblem(numObs: Int, expected: Array[Double]) {
    val problem = new svm_problem
    problem.l = numObs
    problem.y = expected
    problem.x = new SVMNodes(numObs)

    def update(n: Int, node: Array[svm_node]): Unit =
      problem.x(n) = node
  }

  def createNode(dim: Int, x: Array[Double]): Array[svm_node] =
    x.indices./:(new Array[svm_node](dim)){ (newNode, j) =>
      val node = new svm_node
      node.index = j
      node.value = x(j)
      newNode(j) = node
      newNode
    }

  def predictSVM(model: SVMModel, x: Array[Double]): Double =
    svm.svm_predict(model.svmmodel, toNodes(x))

  def crossValidateSVM(
    problem: SVMProblem,
    param: svm_parameter,
    nFolds: Int,
    size: Int
  ): Array[Double] = {

    val target = Array.fill(size)(0.0)
    svm.svm_cross_validation(problem.problem, param, nFolds, target)
    target
  }

  def trainSVM(problem: SVMProblem, param: svm_parameter): svm_model =
    svm.svm_train(problem.problem, param)

  private def toNodes(x: Array[Double]): Array[svm_node] =
    x.view.zipWithIndex./:(new ArrayBuffer[svm_node])((xs, f) => {
      val node = new svm_node
      node.index = f._2
      node.value = f._1
      xs.append(node)
      xs
    }).toArray
}

// ------------------------------  EOF --------------------