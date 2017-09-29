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
package org.scalaml.unsupervised.functionapprox

import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD

import scala.collection.mutable
import scala.io.Source
import scala.util.Random


/**
 * Generator of data in RDD format by randomize an input data file. Random noise with a ratio noise/signal
 * specified by the user is added to the original data.
 * @author Patrick Nicolas
 * @version 0.99.2
 * @param sourceName  Name of the file containing the template dataset
 * @param nTasks Number of concurrent tasks used in processing the data set
 */
final class DataGenerator(sourceName: String, nTasks: Int) {
  require(nTasks > 0, s"DataGenerator nTasks found $nTasks required >0")

  private final val DELIM = " "
  private final val RATIO = 0.05
  var datasetSize: Int = _

  /**
   * Constructor of a noisy data (following a uniform distribution)
   * @param sc spark context
   * @return RDD of (x, y) data set
   */
  def apply(sc: SparkContext): RDD[(Float, Float)] = {
      // See the random noise
    val r = new Random(System.currentTimeMillis + Random.nextLong)
    val src = Source.fromFile(sourceName)
    val input = src.getLines.map(_.split(DELIM))
      ./:(mutable.ArrayBuffer[(Float, Float)]())((buf, xy) => {
      val x = addNoise(xy(0).trim.toFloat, r)
      val y = addNoise(xy(1).trim.toFloat, r)
      buf += ((x, y))
    })
    datasetSize = input.size
    val data_rdd = sc.makeRDD(input, nTasks)
    src.close
    data_rdd
  }
    // Original signal + random noise
  private def addNoise(value: Float, r: Random): Float = value*(1.0 + RATIO*(r.nextDouble - 0.5)).toFloat
}

// -------------------------------------  EOF ----------------------------------------------
