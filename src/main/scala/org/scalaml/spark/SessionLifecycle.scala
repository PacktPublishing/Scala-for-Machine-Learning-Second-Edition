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

import org.apache.spark.SparkConf
import org.apache.spark.sql.{DataFrame, Dataset, Encoders, SparkSession}

/**
 * Class that manages the lifecyle of a spark session
 *
 * @author Patrick Nicolas
 * @see Scala for Machine Learning - 2nd Edition - Apache Spark MLlib
 * @since 0.99.2
 */
private[spark] trait SessionLifeCycle {
  import SessionLifeCycle._
  val sparkSession = SparkSession.builder().appName(AppNameLabel).config(new SparkConf()
    .set("spark.default.parallelism", ParallelismLabel)
    .set("spark.rdd.compress", "true")
    .set("spark.executor.memory", ExecutorMemoryLabel)
    .set("spark.shuffle.spill", "true")
    .set("spark.shuffle.spill.compress", "true")
    .set("spark.io.compression.codec", "lzf")
    .setMaster(MasterLabel))
    .getOrCreate()

  protected def csv2DF(dataFile: String): DataFrame =
    sparkSession.read.option("header", true).option("inferSchema", true).csv(dataFile)

  def sparkContext = sparkSession.sparkContext
  def sqlContext = sparkSession.sqlContext

  def stop: Unit = sparkSession.stop
}

  // Default Session life cycle for testing on local machine
private[spark] object SessionLifeCycle {
  final private val AppNameLabel: String = "SessionLifeCyle"
  final private val ParallelismLabel: String = "4"
  final private val ExecutorMemoryLabel: String = "12g"
  final private val MasterLabel = "local[4]"
}

/**
 * Simple data set generator using an implicit data session
 */
private[spark] object DatasetGenerator {

      // Generation of a dataset of type {Double, Double} with a by-name initialization function
  final def toDSPairDouble(
    numDataPoints: Int
  )(
    generator: Int => (Double, Double)
  )(implicit sessionLifeCycle: SessionLifeCycle): Dataset[(Double, Double)] =
    toDSPairDouble(Seq.tabulate(numDataPoints)(generator(_)))

    // Generation of a dataset of type {Double, Double} from a sequence of same type
  def toDSPairDouble(
    data: Seq[(Double, Double)]
  )(implicit sessionLifeCycle: SessionLifeCycle): Dataset[(Double, Double)] = {
    import sessionLifeCycle.sparkSession.implicits._
    data.toDS()
  }

    // Generation of a dataset of type Double
  def toDSDouble(data: Seq[Double])(implicit sessionLifeCycle: SessionLifeCycle): Dataset[Double] = {
    import sessionLifeCycle.sparkSession.implicits._
    data.toDS()
  }

    // Generation of a dataset of type Int
  def toDSInt(data: Seq[Int])(implicit sessionLifeCycle: SessionLifeCycle): Dataset[Int] = {
    import sessionLifeCycle.sparkSession.implicits._
    data.toDS()
  }
}

// --------------------------  EOF ----------------------------------------------