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
package org.scalaml.supervised

import scala.util.Try
import org.scalaml.Predef._
import org.scalaml.core.ITransform

/**
 * Trait that defined the interface to supervised learning algorithm.
 * The trait requires developers to create a validation routine for parameterized
 * multidimensional time series of tuple (observation, class label).
 * @author Patrick Nicolas
 * @tparam T Type of elements in the time series
 * @tparam V Type of element in the labeled data
 * @see Scala for Machine Learning
 * @version 0.99.2
 */
private[scalaml] trait Supervised[T, V] {
  self: ITransform[T, V] =>
  /**
   * validation method for supervised learning algorithm
   * @param xt parameterized multidimensional time series of tuple (observation, class label)
   * @param expected values or index of the class that contains the true positive labels
   * @return F1 measure
   */
  def validate(xt: Vector[T], expected: Vector[V]): Try[Double]
  def crossValidation: Option[Features] = None
}

// --------------------------------  EOF ------------------------------------------