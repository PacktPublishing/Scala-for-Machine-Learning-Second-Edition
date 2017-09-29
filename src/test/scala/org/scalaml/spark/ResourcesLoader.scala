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
package org.scalaml.spark

import org.apache.spark.sql.DataFrame

sealed trait Extractor {
  protected[this] val delimiter: String
  def extract(line: String): Array[String]
}

/**
  * CSV extractor for loader
  */
final class CSVExtractor extends Extractor {
  override protected[this] val delimiter: String = ","
  override def extract(line: String): Array[String] = line.split(delimiter)
}

private[spark] object ResourcesLoader {
  import scala.io.Source

  type FieldsSet = Iterator[Array[String]]
  final def loadFromLocal(filename: String, extractor: Extractor): FieldsSet = {
    val src = Source.fromFile(filename)
    val lines = src.getLines().map(extractor.extract(_))
    src.close()
    lines
  }

  final def loadFromHDFS(pathname: String)(implicit sessionLifeCycle: SessionLifeCycle): DataFrame = {
    import sessionLifeCycle.sparkSession.implicits._
    sessionLifeCycle.sparkContext.textFile(pathname).toDF
  }

  final def getPath(filename: String): Option[String] = Option(getClass.getResource(filename).getPath)
}

