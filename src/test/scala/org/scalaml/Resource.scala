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

import scala.util.Try

/**
  * Generic trait to manage resources, files used in unit tests
  * @version 0.99.2
  */
private[scalaml] trait Resource {
  final val resPath = "/data/"
  final val textPath = "/text/"

  final def getPath(fileName: String): Try[String] = Try(getClass().getResource(s"${resPath}$fileName").getPath)
  final def getTextPath(fileName: String): Try[String] = Try(getClass().getResource(s"${textPath}$fileName").getPath)

}


// --------------------------------  EOF -------------------------------------------------
