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
package org.scalaml.util

import org.apache.log4j.Logger
import scala.util.{Failure, Success, Try}

/**
 *
 * Singleton that defines the implicit classes that extends Scala standard library classes
 * such as Try, String...
 *
 * @author Patrick Nicolas
 * @since 0.98 December 21, 2014
 *  @version 0.99.2
 */

private[scalaml] object LoggingUtils {
  private val logger = Logger.getLogger("Utils")

  /**
   * Implicit class to extends Try monad
   *
   * @param _try Try instance for which the method are added.
   */
  implicit class TryToOption[T](_try: Try[T]) {

    /**
     * Method that converts a Try monad to an Option with a comment string.
     *
     * @param str Description included in the conversion.
     * @param _logger reference ot logger
     * @return a parameterized option.
     */
    def _toOption(str: String, _logger: Logger): Option[T] = _try match {
      case Success(res) => Some(res)
      case Failure(e) => DisplayUtils.none(s"$str", _logger, e)
    }

    def toOption(str: String)(_logger: Logger): Option[T] = _try match {
      case Success(res) => Some(res)
      case Failure(e) => DisplayUtils.none(s"$str", _logger, e)
    }
    def toOption(_logger: Logger): Option[T] = _try match {
      case Success(res) => Some(res)
      case Failure(e) => DisplayUtils.none(s"Error", _logger, e)
    }

    def _toBoolean(str: String): Boolean = _try match {
      case Success(res) => true
      case Failure(e) => DisplayUtils.error(s"$str", logger, e); false
    }

    def toInt(str: String, n: Int): Int = _try match {
      case Success(res) => n
      case Failure(e) => DisplayUtils.error(s"$str", logger, e)
    }

    /**
     * Method that converts a Try monad to a string with a comment string.
     *
     * @param str Description included in the conversion.
     * @return an error string.
     */
    def _toString(str: String): String = _try match {
      case Success(_try) => _try.toString
      case Failure(e) => "$str ${e.toString}"
    }

    private def failureHandler(str: String, e: Throwable): Int = e match {
      case e: MatchError =>
        DisplayUtils.error(s"$str ${e.getMessage()} caused by ${e.getCause.toString}", logger)
      case _ =>
        DisplayUtils.error(s"$str ${e.toString}", logger)
    }
  }

  trait Monitor[T] {
    import org.scalaml.plots.{Legend, LightPlotTheme, LinePlot}
    import scala.collection.mutable.{HashMap, ArrayBuffer}

    protected val logger: Logger

    private[this] lazy val _counters = HashMap[String, ArrayBuffer[T]]()

    final def counters(key: String): ArrayBuffer[T] = _counters.getOrElse(key, ArrayBuffer.empty[T])

    def count(key: String, value: T): Unit = {
      val buffer: ArrayBuffer[T] = _counters.getOrElse(key, ArrayBuffer.empty[T])
      buffer.append(value)
      _counters.put(key, buffer)
    }

    def dump: String = _counters.map { case (k, v) => s"$k -> ${v.mkString(", ")}" }.mkString("\n")

    final def display(key: String, legend: Legend)(implicit f: T => Double): Boolean =
      LinePlot.display(counters(key: String).map(_.toDouble).toArray, legend, new LightPlotTheme)

    final def display(keys: List[String], legend: Legend)(implicit f: T => Double): Boolean = {
      val isCount = _counters.nonEmpty
      if (isCount) {
        val counts = keys.map(key => (counters(key).map(_.toDouble).toVector, key))
        LinePlot.display(counts, legend, new LightPlotTheme)
      }
      isCount
    }

    final def keySet: Iterable[String] = _counters.keySet

    def show(msg: String): Int = DisplayUtils.show(msg, logger)
    def error(msg: String): Int = DisplayUtils.error(msg, logger)
    def error(msg: String, e: Throwable): Int = DisplayUtils.error(msg, logger, e)
    def none(msg: String) = DisplayUtils.none(msg, logger)
    def none(msg: String, e: Throwable) = DisplayUtils.none(msg, logger, e)
  }
}

// ---------------------------------  EOF ---------------------------------------------