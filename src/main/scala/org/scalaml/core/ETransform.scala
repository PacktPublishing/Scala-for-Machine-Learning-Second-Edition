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
package org.scalaml.core

import org.scalaml.core.Design.Config
import scala.language.higherKinds
import scala.util.Try

/**
  * Define an explicit monadic data transformation. An explicit transformation relies on
  * a configuration (or context) to execute
  * @author Patrick Nicolas
  * @version 0.99.2
  * @param config  Configuration (or context)
  * @tparam T  Type of input to the data transformation
  * @tparam A  Type of output of the data transformation
  * @see Scale for Machine Learning Chapter 2 "Data Pipelines"
  */
private[scalaml] abstract class ETransform[T, A](val config: Config) extends ITransform[T, A] {
  self =>
  override def map[B](f: A => B): ETransform[T, B] = new ETransform[T, B](config) {
    override def |> : PartialFunction[T, Try[B]] = ???
  }

  def flatMap[B](f: A => ETransform[T, B]): ETransform[T, B] = new ETransform[T, B](config) {
    override def |> : PartialFunction[T, Try[B]] = ???
  }

  def compose[B](tr: ETransform[A, B]): ETransform[T, B] = new ETransform[T, B](config) {
    override def |> : PartialFunction[T, Try[B]] = ???
  }
}

// -------------------------------  EOF -----------------------------------