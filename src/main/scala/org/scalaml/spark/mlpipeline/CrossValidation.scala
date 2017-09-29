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

import org.apache.spark.ml.evaluation.{BinaryClassificationEvaluator, Evaluator}
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.tuning.{CrossValidator, CrossValidatorModel}
import org.apache.spark.ml.{Model, Pipeline, PipelineStage}
import org.apache.spark.sql._


/**
  * trait that define the cross-validation as a specialized ModelEstimator. The inheritance
  * relationship enforces the dependency of the cross validation that is applied to this
  * model estimator
  *
  * Note: The type parameterization is implemented through F-bound polymorphism.
  *
  * @author Patrick Nicolas
  * @since 0.99.2  Dec 11, 2016
  * @see Scala for Machine Learning 2nd Edition - Chap 17 - Apache Spark MLlib
  * @tparam T Type of the model to be cross-validated
  */
private[spark] trait CrossValidation[T <: Model[T]] extends ModelEstimator[T] {
  /**
    * Number of folds used in the cross validation
    */
  protected[this] val numFolds: Int

  /**
    * @param grid Grid used in the plan for cross validation model
    * @param stages Sequence of transform or stages in the pipeline
    * @return A cross validation model
    */
  @throws(classOf[IllegalArgumentException])
  protected def apply(
    trainDf: DataFrame,
    stages: Array[PipelineStage],
    grid: Array[ParamMap]
  ): CrossValidatorModel = {
    require(stages.size > 0, "Cannot cross-validate pipeline without stages")
    require(grid.size > 0, "Cannot cross-validate with undefined grid")

    val pipeline = new Pipeline().setStages(stages ++ Array[PipelineStage](estimator))
    new CrossValidator()
      .setEstimator(pipeline)
      .setEstimatorParamMaps(grid)
      .setEvaluator(new BinaryClassificationEvaluator)
      .setNumFolds(numFolds)
      .fit(trainDf)
  }

  protected def evaluate(
    trainDf: DataFrame,
    stages: Array[PipelineStage],
    grid: Array[ParamMap]
  ): Evaluator = this(trainDf, stages, grid).getEvaluator
}
