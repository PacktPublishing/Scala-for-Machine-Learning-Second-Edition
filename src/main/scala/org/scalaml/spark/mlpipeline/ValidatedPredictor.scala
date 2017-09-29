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

import org.apache.spark.ml.evaluation.Evaluator
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.tuning.CrossValidatorModel
import org.apache.spark.ml.{Estimator, PipelineModel, Model}
import org.apache.spark.sql._


/**
  * Combination of a prediction and a cross-validation models. The pipeline is
  * executed with an estimator which is then fed into a cross validator
  * @author Patrick Nicolas
  * @version 0.99.2
  *
  * @param estimate Estimator used in the pipeline
  * @param cols  List of features or columns in the data frame
  * @param trainFile training set loaded from a file
  * @param numFolds Number of fold (default = 2)
  * @tparam T Type of the model (logistic regression, SVM, ...)
  * @see Scala for Machine Learning Chapter 17 Apache Spark MLlib
  */
@throws(classOf[IllegalArgumentException])
final private[spark] class ValidatedPredictor[T <: Model[T]](
    estimate: Estimator[T],
    cols: Array[String],
    trainFile: String,
    override val numFolds: Int = 2
) extends Predictor[T](estimate, cols, trainFile) with CrossValidation[T] {
  require(cols.size > 0, "Cannot predict with undefined attributes")

  /**
    * Generate a pipeline model, including the logisitic regression as estimator
    * @return a trained pipeline model
    */
  def apply(): PipelineModel = this(trainDf, stages)

  /**
    * Generate a CrossValidator model
    * @param paramGrid Parameters used in the test grid
    * @return CrossValidator predictor model
    */
  @throws(classOf[IllegalArgumentException])
  def apply(paramGrid: Array[ParamMap]): CrossValidatorModel = {
    require(paramGrid.size > 0, "Cannot validate with undefined grid")
    this(trainDf, stages, paramGrid)
  }


  /**
    * Execute a classifier predictive model following a grid test
    * @param grid List of grid parameters
    * @param testSet test set used in the validation
    * @return Prediction as a dataframe
    */
  @throws(classOf[IllegalArgumentException])
  final def classify(grid: Array[ParamMap], testSet: String): DataFrame =
    this(trainDf, stages, grid).transform(csv2DF(testSet))

  /**
    *
    * @param grid
    * @return
    */
  final def evaluate(grid: Array[ParamMap]): Evaluator = evaluate(trainDf, stages, grid)

  /**
    *
    * @return
    */
  final def trainingWithSummary: Option[(Double, Double)] = trainWithSummary(trainDf, stages)
}

// ----------------------------------------  EOF ----------------------------------------------------------