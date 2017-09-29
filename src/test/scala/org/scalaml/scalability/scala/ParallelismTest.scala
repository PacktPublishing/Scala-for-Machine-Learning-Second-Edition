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
package org.scalaml.scalability.scala

import org.scalaml.Logging
import org.scalatest.{FlatSpec, Matchers}

/**
  * '''Purpose''': Singleton to evaluate the performance of Scala parallel arrays
  * and maps.
  *
  * @author Patrick Nicolas
  * @note Scala for Machine Learning Chapter 16 Scalable frameworks/Scala/Parallel collections
  */
final class ParallelismTest extends FlatSpec with Matchers with Logging {
  import scala.collection.mutable.HashMap
  import scala.collection.parallel.mutable.{ParArray, ParHashMap}
  import scala.util.Random

  protected[this] val name: String = "Scala parallel collections"

  final private val SZ = 100000
  final private val NUM_TASKS = 8
  final private val evalRange = Range(1, NUM_TASKS)
  final private val TIMES = 20

  // Arbitrary map function
  final val mapF = (x: Double) => Math.sin(x * 0.01) + Math.exp(-x)

  // Arbitrary filter function
  final val filterF = (x: Double) => x > 0.8

  // Arbitrary reduce function
  final val reduceF = (x: Double, y: Double) => (x + y) * x


  it should s"$name: arrays" in {
    show(s"Evaluation of arrays")

    // Generate random vector for both the non-parallel and parallel array
    val data = Array.fill(SZ)(Random.nextDouble)
    val pData = ParArray.fill(SZ)(Random.nextDouble)

    // Initialized and execute the benchmark for the parallel array
    val benchmark = new ParallelArray[Double](data, pData, TIMES)

    val ratios = new Array[Double](NUM_TASKS)
    evalRange.foreach(n => ratios.update(n, benchmark.map(mapF)(n)))
    val resultMap = ratios.tail
    resultMap.sum / resultMap.size < 1.0 should be(true)
    display(resultMap, "ParArray.map")

    evalRange.foreach(n => ratios.update(n, benchmark.filter(filterF)(n)))
    val resultfilter = ratios.tail
    resultfilter.sum / resultfilter.size < 1.0 should be(true)
    display(resultfilter, "ParArray.filter")
  }

  it should s"$name: maps" in {
    show("Evaluation of maps")

    val mapData = new HashMap[Int, Double]
    Range(0, SZ).foreach(n => mapData.put(n, Random.nextDouble))
    val parMapData = new ParHashMap[Int, Double]
    Range(0, SZ).foreach(n => parMapData.put(n, Random.nextDouble))

    // Initialized and execute the benchmark for the parallel map
    val benchmark = new ParallelMap[Double](mapData.toMap, parMapData, TIMES)

    val ratios = new Array[Double](NUM_TASKS)
    evalRange.foreach(n => ratios.update(n, benchmark.map(mapF)(n)))
    val resultMap = ratios.tail
    resultMap.sum / resultMap.size < 1.0 should be(true)
    display(resultMap, "ParMap.map")

    evalRange.foreach(n => ratios.update(n, benchmark.filter(filterF)(n)))
    val resultfilter = ratios.tail
    resultfilter.sum / resultfilter.size < 1.0 should be(true)
  }


  private def display(x: Array[Double], label: String): Unit = {
    import org.scalaml.plots.{Legend, LightPlotTheme, LinePlot}

    val labels = Legend(
      name,
      "Scala parallel collections",
      s"Scala parallel computation for $label",
      "Relative timing"
    )
    LinePlot.display(x.toVector, labels, new LightPlotTheme)
  }
}

// -------------------------------------------  EOF --------------------------------------------------