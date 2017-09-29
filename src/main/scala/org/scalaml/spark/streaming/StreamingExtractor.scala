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
import org.apache.spark.streaming.dstream.InputDStream
import org.apache.spark.streaming.{Duration, Milliseconds, Seconds, StreamingContext}

import scala.reflect.ClassTag

/**
  * Streaming processor that extract an input of type T to a sequence of elements of type U
  *
  * @author Patrick Nicolas
  * @since 0.99.2
  * @tparam T  Type of input to stream
  * @tparam U  Type of elements of sequence output by the online/streaming extractor
  */
private[spark] class StreamingExtractor[T, U: ClassTag](sparkContext: SparkContext) extends Serializable {
  /**
    * Extractor data from an input stream
    * @param inputStream type parameterized input stream
    * @param f transform to be applied to the data extracted from a sequence of files.
    * @param extractor extracting function
    */
  def extract(inputStream: InputDStream[T], f: RDD[U] => Unit, extractor: T => U): Unit =
    inputStream.map(extractor(_)).foreachRDD(f(_))


  /**
    * Wraps the processing of text file through the Spark context
    * @param fileName file to be loaded and processed
    * @return RDD of strings
    */
  final def textFile(fileName: String): RDD[String] = sparkContext.textFile(fileName)
}

// -------------------------------------------  EOF -----------------------------------------------