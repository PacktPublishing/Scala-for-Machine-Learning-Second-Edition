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

import org.apache.spark.ml.classification.{LogisticRegression, LogisticRegressionModel}
import org.apache.spark.ml.param.ParamMap
import org.scalaml.Logging
import org.scalaml.spark.ResourcesLoader
import org.scalatest.{FlatSpec, Matchers}

final class MLPipelineTest extends FlatSpec with Matchers with Logging {
  protected val name = "Spark ML pipeline"

  final val trainFile = "/data/spark/mlpipeline_training.csv"
  final val testFile = "/data/spark/mlpipeline_test.csv"


  final val columns = Array[String]("date", "asset", "region", "agent")

  it should s"$name simple predictor" in {
    show(s"$name simple predictor")

    (for {
      trainPath <- ResourcesLoader.getPath(trainFile)
      testPath <- ResourcesLoader.getPath(testFile)
    } yield {
      val predictor = new SimplePredictor[LogisticRegressionModel](
        new LogisticRegression().setMaxIter(5).setRegParam(0.1),
        columns,
        trainPath
      )

      (predictor, predictor.classify(predictor(), testPath))
    }).map {
      case (predictor, output) => {
        output.printSchema
        val predictedValues = output.select("prediction").collect.map(_.getDouble(0))
        output.show

        predictor.stop
        predictedValues(0)
      } should be(0.0)
    }
  }

  /**
    * Cross-validation on a Logistic regression model
    */
  it should s"$name cross validation" in {
    show(s"$name cross validation")

    (for {
      trainPath <- ResourcesLoader.getPath(trainFile)
      testPath <- ResourcesLoader.getPath(testFile)
    } yield {
      val lr = new LogisticRegression().setMaxIter(5).setRegParam(0.1)
      val paramsMap = new ParamMap().put(lr.maxIter -> 30).put(lr.regParam -> 0.1)
      val validator = new ValidatedPredictor[LogisticRegressionModel](lr, columns, trainPath)

      val (f1, auROC) = validator.trainingWithSummary.getOrElse((Double.NaN, Double.NaN))
      println(s"F1-measure = ${f1} auROC = ${auROC}")
      validator.stop
      f1 should be(0.025 +- 0.005)
      auROC should be(0.600 +- 0.005)
    })
  }
}

// --------------------------------  EOF ---------------------------------------------