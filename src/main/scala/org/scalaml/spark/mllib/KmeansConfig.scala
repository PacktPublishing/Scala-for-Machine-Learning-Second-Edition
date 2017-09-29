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

import org.apache.spark.mllib.clustering.KMeans

/**
  * Define the configuration of the Spark KMeans wrapper.
  * @constructor Create a configuration for the Spark K-means algorithm.
  * @param K Number of clusters used in Spark KMeans
  * @param maxNumIters  Maximum number of iterations allowed for Spark KMeans
  * @param numRuns  Number of runs to be executed by Spark KMeans.
  * @throws IllegalArgumentException if any of the parameters is out of range
  * @author Patrick Nicolas
  * @since 0.98.2 April, 2, 2014
  * @note Scala for Machine Learning Chapter 17 Apache Spark MLlib
  */
final private[spark] class KmeansConfig(K: Int, maxNumIters: Int, numRuns: Int = 1) {
  import KmeansConfig._

  check(K, maxNumIters, numRuns)

  /**
    * Reference to MLlib KMeans class that is initialized with the class parameters
    */
  val kmeans = {
    val kmeans = new KMeans
    kmeans.setK(K)
    kmeans.setMaxIterations(maxNumIters)
    kmeans.setRuns(numRuns)
    kmeans
  }
}

/**
  * Companion object for the Spark K-means configuration class. The singleton
  * defines the constructors and validate its parameters.
  *
  * @author Patrick Nicolas
  * @since 0.98.2 April 2, 2014
  * @see Scala for Machine Learning Chapter 17 Apache Spark MLlib
  */
private[spark] object KmeansConfig {
  private val MAX_NUM_CLUSTERS = 500
  private val MAX_NUM_ITERS = 250
  private val MAX_NUM_RUNS = 500

  private def check(K: Int, maxNumIters: Int, numRuns: Int): Unit = {
    require(K > 0 && K < MAX_NUM_CLUSTERS, "Number of clusters K $K is out of range")
    require(
      maxNumIters > 0 && maxNumIters < MAX_NUM_ITERS,
      s"Maximum number of iterations $maxNumIters is out of range"
    )
    require(
      numRuns > 0 && numRuns < MAX_NUM_RUNS,
      s"Maximum number of runs for K-means $numRuns is out of range"
    )
  }
}

// --------------------------------------  EOF ---------------------------------------------------