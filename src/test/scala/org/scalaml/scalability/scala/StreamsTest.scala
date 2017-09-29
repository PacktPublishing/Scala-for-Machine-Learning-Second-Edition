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

import java.lang.ref._

import org.apache.log4j.Logger
import org.scalaml.Logging
import org.scalaml.Predef._
import org.scalatest.{FlatSpec, Matchers}

import scala.math._

case class DataPoint(x: DblVec, y: Double)

/**
  * Generic loss function for a model using margin f and weights
  * @param f Function to compute margin
  * @param weights Model weights
  * @param dataSize  Size of the input dataset
  */
class LossFunction(f: (DblVec, DblVec) => Double, weights: DblVec, dataSize: Int) {

  var nElements = 0
  private val logger = Logger.getLogger("LossFunction")
  private val STEP = dataSize / 10

  def compute(stream: () => WeakReference[Stream[DataPoint]]): Double = compute(stream().get, 0.0)

  @scala.annotation.tailrec
  private def compute(stream: Stream[DataPoint], loss: Double): Double = {
    if (nElements >= dataSize)
      loss
    else {
      val step = if (nElements + STEP > dataSize) dataSize - nElements else STEP
      nElements += step
      val newLoss = _loss(stream.take(step).toList)
      compute(stream.drop(STEP), loss + newLoss)
    }
  }

  def _loss(xs: List[DataPoint]): Double = xs.map(dp => {
    val z = dp.y - f(weights, dp.x)
    z * z
  }).reduce(_ + _)
}

/**
  * '''Purpose''': Singleton to um(z+illustrate Scala streams
  *
  * @author Patrick Nicolas
  * @see Scala for Machine Learning Chapter 16 Parallelism/ Scala streams
  */
final class StreamsTest extends FlatSpec with Matchers with Logging {
  import scala.util.Random

  protected[this] val name = "Scala streams"

  it should s"$name huge list" in {
    show(s"$name huge list")

    val input = (0 until 1000000000).toStream
    input(10) should be(10)
  }

  it should s"$name recursion" in {
    show(s"$name recursion")

    def mean(strm: => Stream[Double]): Double = {
      @scala.annotation.tailrec
      def mean(z: Double, count: Int, strm: Stream[Double]): (Double, Int) =
        if (strm.isEmpty)
          (z, count)
        else
          mean((1.0 - 1.0 / count) * z + strm.head / count, count + 1, strm.tail)
      mean(0.0, 1, strm)._1
    }

    val input = List[Double](2.0, 5.0, 3.5, 2.0, 5.7, 1.0, 8.0)
    val ave: Double = mean(input.toStream)
    ave should be(3.88 +- 0.05)
  }

  it should s"$name with recycled memory blocks" in {
    show("$name with recycled memory blocks")

    type DblVec = Vector[Double]
    val DATASIZE = 20000

    val dot = (s: Double, xy: (Double, Double)) => s + xy._1 * xy._2
    val diff = (x: DblVec, y: DblVec) => x.zip(y).aggregate(0.0)(dot, _ + _)

    val weights = Vector[Double](0.5, 0.7)
    val lossFunction = new LossFunction(diff, weights, DATASIZE)

    // Create a stream of weak references to 10 stream segments of size DATESIZE/10
    val stream = () => new WeakReference(
      Stream.tabulate(DATASIZE)(n =>
        DataPoint(
          Vector[Double](n.toDouble, n * (n.toDouble)),
          n.toDouble * weights(0) + n * (n.toDouble) * weights(1) + 0.1 * Random.nextDouble
        ))
    )
    // Compute a simple distance using the dot product
    val totalLoss = sqrt(lossFunction.compute(stream))
    show(s"$name totalLoss ${totalLoss / DATASIZE}")

    val averageLoss = totalLoss / DATASIZE
    averageLoss should be(0.0 +- 0.001)
  }
}

// --------------------------  EOF --------------------------------