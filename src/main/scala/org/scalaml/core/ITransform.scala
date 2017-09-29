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

import scala.language.higherKinds
import scala.util.Try

/**
 * Monadic implementation of a data transformation from input of type T to
 * an output of type Try[A]
 * {{{
 *   		data: Vector[T] => implicit model
 *      t: T -> | implicit model } -> u: U
 * }}}
 * The Partial function avoid the need define all the condition on the type and value of
 * the input inj sub-classes
 * A MatchErr is thrown if the value or type of the input value is not supported.
 * @author Patrick Nicolas
 * @version 0.99.2
 * @tparam T  Type of input to the data transformation
 * @tparam A  Type of output of the data transformation
 * @see Scale for Machine Learning Chapter 2 "Data Pipelines"
 */
private[scalaml] trait ITransform[T, A] {
  self =>

  /**
   * Declaration of the actual data transformation that take an input of type T
   * @return A partial function that implement the conversion of data element T => Try[A]
   */
  def |> : PartialFunction[T, Try[A]]
  /**
   * Implementation of the map method
   *
   * @tparam B type of the output of morphism on element of a data
   * @param  f function that converts from type T to type U
   */
  def map[B](f: A => B): ITransform[T, B] = new ITransform[T, B] {
    override def |> : PartialFunction[T, Try[B]] =
      new PartialFunction[T, Try[B]] {
        override def isDefinedAt(t: T): Boolean = self.|>.isDefinedAt(t)
        override def apply(t: T): Try[B] = self.|>(t).map(f)
      }
  }

  /**
   * Implementation of flatMap
   *
   * @tparam B type of the output of morphism on element of a data
   * @param f function that converts from type T to a monadic container of type U
   */
  def flatMap[B](f: A => ITransform[T, B]): ITransform[T, B] = new ITransform[T, B] {
    override def |> : PartialFunction[T, Try[B]] =
      new PartialFunction[T, Try[B]] {
        override def isDefinedAt(t: T): Boolean = self.|>.isDefinedAt(t)
        override def apply(t: T): Try[B] = self.|>(t).flatMap(f(_).|>(t))
      }
  }

  def compose[B](tr: ITransform[A, B]): ITransform[T, B] = new ITransform[T, B] {
    override def |> : PartialFunction[T, Try[B]] =
      new PartialFunction[T, Try[B]] {
        override def isDefinedAt(t: T): Boolean = self.|>.isDefinedAt(t) && tr.|>.isDefinedAt(self.|>(t).get)
        override def apply(t: T): Try[B] = tr.|>(self.|>(t).get)
      }
  }
}

// -------------------------------  EOF -----------------------------------