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
package org.scalaml.spark.streaming

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.streaming.{Duration, Seconds, StreamingContext}

/**
  * Class that manages the lifecucle of the Apache Spark streaming context
  * @author Patrick Nicolas
  * @version 0.99.2
  */
private[spark] trait StreamingLifeCycle {
  val timeOut: Long
  protected[this] val batchDuration: Duration = Seconds(1)
  val streamingContext = new StreamingContext(
    new SparkConf()
      .setMaster("local[4]")
      .setAppName("StreamingStats")
      .set("spark.default.parallelism", "4")
      .set("spark.rdd.compress", "true")
      .set("spark.executor.memory", "8g")
      .set("spark.shuffle.spill", "true")
      .set("spark.shuffle.spill.compress", "true")
      .set("spark.io.compression.codec", "lzf"),
    Seconds(2)
  )

  def sparkContext: SparkContext = streamingContext.sparkContext

  def start: Unit = streamingContext.start
  def terminate: Unit = {
    streamingContext.stop(true, true)
    streamingContext.awaitTerminationOrTimeout(timeOut)
  }
}

// ----------------------------------------------  EOF --------------------------------