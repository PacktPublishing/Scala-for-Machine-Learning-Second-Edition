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

/**
  * Predictor class that is defined as a Model factory for a model of type T with
  * a model estimator of identical type
  * @author Patrick Nicolas
  * @version 0.99.2
  *
  * @param estimate Estimator used to generate the pipeline model
  * @param cols Array of column names
  * @param trainFile file
  * @tparam T Type of the classifier (Logistic regression, SVM,...)
  * @see Scala for Machine Learning Chapter 17 Apache Spark MLlib
  */
@throws(classOf[IllegalArgumentException])
private[spark] final class SimplePredictor[T <: Model[T]](
    estimate: Estimator[T],
    cols: Array[String],
    trainFile: String
) extends Predictor[T](estimate, cols, trainFile) with ModelEstimator[T] {
  require(cols.size > 0, "Cannot predict with undefined attributes")

  /**
    * Constructor for a pipeline model from a resource file
    * @return Pipeline model
    */
  def apply(): PipelineModel = this(trainDf, stages)
}

// -------------------------------------------------  EOF --------------------------------------------

