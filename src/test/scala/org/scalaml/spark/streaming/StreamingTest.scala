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

import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import org.apache.spark.streaming.StreamingContext
import org.scalaml.{Logging, Resource}
import org.scalatest.{BeforeAndAfterAll, FlatSpec, Matchers}

import scala.collection.mutable

case class Account(date: String, asset: String, region: String, agent: String, label: String)

object Account {
  def apply(line: String) : Account = {
    val fields = line.split(",")
    Account(fields(0), fields(1), fields(2), fields(3), fields(4))
  }
}

/**
  * Evaluation of Apache Spark streaming for the continuous extraction of data.
  */
final class StreamingTest extends FlatSpec with Matchers with Logging with Resource {
  protected[this] val name = "Apache Spark Streaming"

    // Define the configuration of the streamer life cycle that initializes
    // the Spark and Streaming contexts
  private var active = false


  final val fileNames = Array[String](
    "streaming_input.csv", "streaming_input2.csv"
  )


  it should s"$name data extraction without checkpoint" in {
    show("Data extraction without checkpoint")

    val streamer: StreamingLifeCycle = new StreamingLifeCycle {
      override val timeOut: Long = 10L
    }
    implicit val streamingContext: StreamingContext = streamer.streamingContext
    implicit val sparkContext: SparkContext = streamer.sparkContext

    extractData(sparkContext, streamingContext)

    streamingContext.start
    streamingContext.stop(true, true)
    Thread.sleep(2000)
    streamingContext.awaitTerminationOrTimeout(streamer.timeOut)
  }


  it should s"$name data extraction with check point" in {
    show("Data extraction with check point")

    val streamer: StreamingLifeCycle = new StreamingLifeCycle {
      override val timeOut: Long = 10L
    }
    implicit val streamingContext: StreamingContext = streamer.streamingContext
    implicit val sparkContext: SparkContext = streamer.sparkContext

    extractData(sparkContext, streamingContext)

    val relativePath = "spark/checkpoint"
    val path: String = getPath(relativePath).getOrElse(".")
    val newStreamingContext: StreamingContext = if (fileNames.exists(!new java.io.File(_).exists))
      StreamingContext.getOrCreate(path, () => streamingContext)
    else {
      streamingContext.checkpoint(path)
      streamingContext
    }

    streamingContext.start
    streamingContext.stop(true, true)
    Thread.sleep(2000)
    streamingContext.awaitTerminationOrTimeout(streamer.timeOut)
  }


  private def extractData(sparkContext: SparkContext, streamingContext: StreamingContext): Unit = {
    val extractor = new StreamingExtractor[String, Account](sparkContext)

    // Build the sequence of RDDs from the sequence of filenames
    val input = fileNames.map(fileName => {
      val relativePath = s"/spark/$fileName"
      val path: String = getPath(relativePath).getOrElse(".")
      extractor.textFile(path)
    })./:(mutable.Queue[RDD[String]]())((qStream, rdd) => qStream += rdd)

    val rddStream = streamingContext.queueStream[String](input, true)

    extractor.extract(rddStream, (rddStr: RDD[Account]) => {
      val content = rddStr.collect.mkString("\n")
      show(s"$name content extracted through streaming $content")
    }, (str: String) => Account(str))
  }

}
