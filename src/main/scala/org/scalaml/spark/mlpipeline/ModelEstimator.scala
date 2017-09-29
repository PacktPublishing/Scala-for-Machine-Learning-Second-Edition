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


import org.apache.spark.ml.classification.{BinaryLogisticRegressionSummary, LogisticRegressionModel}
import org.apache.spark.ml._
import org.apache.spark.sql
import sql._

/**
  * Model estimator used for the classification. The class includes the computation
  * of quality measure (F1 and area under the ROC)
  * @author Patrick Nicolas
  * @version 0.99.2
  * @tparam T type of the model to be estimated
  */
private[spark] trait ModelEstimator[T <: Model[T]] {
  protected[this] val estimator: Estimator[T]
  /**
    * Set up and execute the pipeline
    * @param trainDf Data frame containint the training set
    * @param stages The different stages or transforms in the pipeline
    * @return Pipeline predictive model
    */
  @throws(classOf[IllegalArgumentException])
  def apply(trainDf: DataFrame, stages: Array[PipelineStage]): PipelineModel = {
    require(stages.size > 0, "Cannot process a pipeline without stages")

    trainDf.printSchema // For debugging purpose - schema
    val pipeline = new Pipeline().setStages(stages ++ Array[PipelineStage](estimator))
    pipeline.fit(trainDf)
  }

  /**
    * Extraction of the summary information regarding ROC for the binary classifier
    * @param trainDf Data frame containint the training set
    * @param stages The different stages or transforms in the pipeline
    * @return Pair of f1-score and area under the ROC.
    */
  @throws(classOf[IllegalArgumentException])
  final def trainWithSummary(
    trainDf: DataFrame,
    stages: Array[PipelineStage]
  ): Option[(Double, Double)] = {
    require(stages.size > 0, "Cannot process a pipeline without stages")

    // Print the training set data frame
    trainDf.printSchema

    this(trainDf, stages).stages.last match {
      case lrModel: LogisticRegressionModel =>
        val binarySummary = lrModel.summary.asInstanceOf[BinaryLogisticRegressionSummary]

        // Set the model threshold to maximize F-Measure
        val f1: Double = binarySummary.fMeasureByThreshold.select("F-Measure").head.getDouble(0)
        Some(f1, binarySummary.areaUnderROC)
      case _ => None
    }
  }
}

// ------------------------------  EOF --------------------------------------------------------