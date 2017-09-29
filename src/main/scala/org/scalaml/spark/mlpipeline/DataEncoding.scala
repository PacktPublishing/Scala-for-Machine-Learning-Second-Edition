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
package org.scalaml.spark.mlpipeline

import org.apache.spark.ml.PipelineStage
import org.apache.spark.ml.feature.{OneHotEncoder, StringIndexer, VectorAssembler}

/**
  * Feature encoding using Apache Spark ML Pipelines
  * @author Patick Nicolas
  * @version 0.99.2
  */
private[spark] trait DataEncoding {
  protected[this] val colNames: Array[String]
  private[this] lazy val vectorizedColNames = colNames.map(vector(_))

  /**
    * Create a pipeline of Spark transformation related to indexing, encoding and assembling features
    */
  lazy val stages: Array[PipelineStage] =
    colNames.map(col => new StringIndexer().setInputCol(col).setOutputCol(index(col))) ++
      colNames.map(col => new OneHotEncoder().setInputCol(index(col)).setOutputCol(vector(col))) ++
      Array[PipelineStage](new VectorAssembler().setInputCols(vectorizedColNames).setOutputCol("features"))

  private def index(colName: String): String = s"${colName}Index"
  private def vector(colName: String): String = s"${colName}Vector"

  override def toString: String = stages.mkString("/n")
}

// -------------------------------------------  EOF -------------------------------------------