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
package org.scalaml.spark.mllib

import org.apache.log4j.Logger
import org.apache.spark.SparkContext
import org.apache.spark.mllib.clustering.KMeansModel
import org.apache.spark.mllib.linalg.DenseVector
import org.scalaml.core.ITransform
import org.scalaml.Predef._
import org.scalaml.util.FormatUtils

import scala.annotation.implicitNotFound
import scala.util.Try

/**
  * Class wrapper for the Spark KMeans implementation. The model is fully generated through
  * training during instantiation of objects in order to reduce their life-cycle
  *
  * The algorithm implements the default data transformation interface, PipeOperator.
  * @constructor Create a wrapper for the Spark K-means algorithm.
  * @throws IllegalArgumentException if the configuration or the time series is undefined.
  * @param kMeansConfig Configuration of the Spark KMeans
  * @param rddConfig Configuration parameters for the Spark RDD
  * @param xt Time series used for the training of the Spark KMeans
  * @param sc  implicit spark context.
  * @author Patrick Nicolas
  * @since 0.98 April 2, 2014
  * @version 0.99.2
  * @see Scala for Machine Learning Chapter 17 Apache Spark MLlib
  */
@implicitNotFound(msg = "SparkKMeans Spark context is not implicitely defined")
final private[spark] class Kmeans(
    kMeansConfig: KmeansConfig,
    rddConfig: RDDConfig,
    xt: Vector[Array[Double]]
)(implicit sc: SparkContext) extends ITransform[Array[Double], Int] {

  import Kmeans._
  check(xt)

  private val logger = Logger.getLogger("KMeans")

  private[this] val model: Option[KMeansModel] = train

  /**
    * Method that classify a new data point in any of the cluster.
    * @throws IllegalArgumentException if the data point is not defined
    * @return the id of the cluster if succeeds, None otherwise.
    */
  override def |> : PartialFunction[Array[Double], Try[Int]] = {
    case x: Array[Double] if (x.nonEmpty && model.isDefined) =>
      Try[Int](model.get.predict(new DenseVector(x)))
  }

  override def toString: String = {
    val header = "K-Means cluster centers from training\nIndex\t\tCentroids\n"
    model.map(_.clusterCenters
      .zipWithIndex
      .map(ctr => s"#${ctr._2}: ${FormatUtils.format(ctr._1.toArray)}\n").mkString("\n"))
      .getOrElse("Model undefined")
  }

  private def train: Option[KMeansModel] =
    Try(kMeansConfig.kmeans.run(RDDSource.convert(xt, rddConfig))).toOption
}

/**
  * Companion object for the Spark K-means class. The singleton
  * defines the constructors and validate its parameters.
  * @author Patrick Nicolas
  * @since April 2, 2014
  * @note Scala for Machine Learning Chapter 17 Apache Spark MLlib
  */
private[spark] object Kmeans {
  /**
    * Default constructor for SparkKMeans class
    * @param config Configuration of the Spark KMeans
    * @param rddConfig Configuration parameters for the Spark RDD
    * @param xt Time series used for the training of the Spark KMeans
    * @param sc  implicit spark context.
    */
  def apply(
    config: KmeansConfig,
    rddConfig: RDDConfig,
    xt: Vector[Array[Double]]
  )(implicit sc: SparkContext): Kmeans = new Kmeans(config, rddConfig, xt)

  /**
    * Default constructor for SparkKMeans class
    * @param config Configuration of the Spark KMeans
    * @param rddConfig Configuration parameters for the Spark RDD
    * @param xt Time series used for the training of the Spark KMeans
    * @param sc  implicit spark context.
    */
  def apply(
    config: KmeansConfig,
    rddConfig: RDDConfig,
    xt: DblMatrix
  )(implicit sc: SparkContext): Kmeans = new Kmeans(config, rddConfig, xt.toVector)

  private def check(xt: Vector[Array[Double]]): Unit = {
    require(!xt.isEmpty, "SparkKMeans.check input time series for Spark K-means is undefined")
  }
}

// --------------------------------------  EOF ---------------------------------------------------