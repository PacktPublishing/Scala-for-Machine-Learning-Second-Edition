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

import org.apache.spark.ml.{Estimator, Model, PipelineModel}
import org.apache.spark.sql._
import org.scalaml.spark.SessionLifeCycle

/**
  * Generic predictive model associated to a specific estimator (ML algorithm) such
  * as SVM or Logistic regression. The subclasses are
  * {{{
  *   Simple predictor
  *   Validated predictor
  * }}}
  * @param estimate  estimator transform
  * @param cols name of columns used
  * @param trainFile name of file containing the observations used in training
  * @tparam T Type of the model (ML algorithm)
  * @author Patrick Nicolas
  * @version 0.99.2
  * @see Scala for Machine Learning Chapter 17 Apache Spark MLlib
  */
abstract class Predictor[T <: Model[T]](
    estimate: Estimator[T],
    cols: Array[String],
    trainFile: String
) extends DataEncoding with SessionLifeCycle {

  override protected[this] val colNames: Array[String] = cols
  protected[this] val trainDf: DataFrame = csv2DF(trainFile)

  /**
    * Estimator used in this ML pipeline
    */
  protected[this] val estimator: Estimator[T] = estimate

  def classify(model: PipelineModel, testSet: DataFrame): DataFrame = model.transform(testSet)
  def classify(model: PipelineModel, testFile: String): DataFrame = model.transform(csv2DF(testFile))
}

// ------------------------------------ EOF --------------------------------------