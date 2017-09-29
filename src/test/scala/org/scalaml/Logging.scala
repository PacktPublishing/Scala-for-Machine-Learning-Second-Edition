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
package org.scalaml

import org.apache.log4j.Logger
import org.scalaml.util.DisplayUtils

private[scalaml] trait Logging {
  protected[this] val name: String
  protected[this] lazy val logger: Logger = Logger.getLogger(s"$name")

  protected def show(description: String): Int = DisplayUtils.show(s"$name $description", logger)

  protected def error(description: String): Int = DisplayUtils.error(s"$name $description", logger)

  protected def error(description: String, e: Throwable): Int = {
    DisplayUtils.error(s"$name $description", logger, e)
    0
  }

  protected def none(description: String): Option[Int] = DisplayUtils.none(s"$name $description", logger)

  /**
   * Handler for MatchErr exception thrown by Partial Functions.
   */
  protected def failureHandler(e: Throwable): Int =
    if (e.getMessage != null) error(s"$name ${e.getMessage} caused by ${e.getCause.toString}")
    else error(s"$name ${e.toString}")
}

