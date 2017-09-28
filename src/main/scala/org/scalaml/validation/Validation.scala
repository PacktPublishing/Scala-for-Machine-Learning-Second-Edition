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
package org.scalaml.validation

import org.scalaml.Predef.Context.ToDouble
import org.scalaml.Predef._

/**
 * Generic trait for the validation method that define a single validation scoring method
 * @author Patrick Nicolas
 * @since 0.98.2 January 29, 2014
 * @version 0.99.2
 * @see Scala for Machine Learning Chapter 2 "Hello World!" / Assessing a model / Validation
 * @see org.scalaml.stats.FValidation
 */
private[scalaml] trait Validation {
  type LabeledData[T] = (Array[T], Int)
  type ValidationType[T] = Vector[LabeledData[T]]
  /**
   * Generic computation of the score of the validation of a classifier or clustering model
   * @return score of the classifier
   */
  def score: Double
}

/**
 * Generic classifier model validation using a F score.
 * @tparam T Type of features that is view bounded to a Double
 * @param labeled  Labeled observations for which the features is array of
 * element of type T and the labeled values are the class index (Int)
 * @author Patrick Nicolas
 * @version 0.99.2
 * @see Scala for Machine Learning Chap 2 ''Hello World!'' / Assessing a model / Validation
 */
abstract private[scalaml] class AValidation[T] protected (
    labeled: Vector[(Array[T], Int)]
) extends Validation {

  def this(expected: Vector[Int], xt: Vector[Array[T]]) = this(xt.zip(expected))

  override def score: Double
}

// --------------------  EOF --------------------------------------------------------