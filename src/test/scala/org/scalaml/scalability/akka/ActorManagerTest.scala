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
package org.scalaml.scalability.akka

import akka.actor.{ActorSystem, Props}
import org.scalaml.Logging
import org.scalaml.Predef.DblVec
import org.scalaml.filtering.dft.DFT
import org.scalaml.scalability.akka.DFTMaster._
import org.scalaml.scalability.akka.message.Start
import org.scalatest.{FlatSpec, Matchers}

import scala.collection._
import scala.util.Random

/**
  * Specialized Akka master actor for the distributed discrete Fourier transform without
  * routing.
  * @constructor Create a master actor for the distributed discrete Fourier transform.
  * @throws IllegalArgumentException if the time series or the partitioner are not defined.
  * @param xt Time series to be processed
  * @param nPartitions Number of partitions used to distributed the input to the discrete Fourier transform
  * @param reducer User defined aggregation and monitoring method
  *
  * @author Patrick Nicolas
  * @since 0.98.2 (June 5, 2014)
  * @version 0.99.2
  * @see Scala for Machine Learning Chapter 16 Scalable frameworks / Akka
  */
protected class DFTMaster(
  xt: DblVec,
  nPartitions: Int,
  reducer: Reducer
) extends Master(xt, DFT[Double].|>, nPartitions)

/**
  * Specialized Akka master actor for the distributed discrete Fourier transform routing.
  * @constructor Create a master actor for the distributed discrete Fourier transform.
  * @throws IllegalArgumentException if the time series or the partitioner are not defined.
  * @param xt Time series to be processed
  * @param reducer Reducing or aggregation method
  * @param nPartitions Number of partitions used in processing the DFT
  *
  * @author Patrick Nicolas
  * @since 0.98.3  (Jun 4, 2015
  * @version 0.99.2
  * @note Scala for Machine Learning Chapter 16 Scalable frameworks / Akka
  */
protected class DFTMasterWithRouter(
  xt: DblVec,
  nPartitions: Int,
  reducer: Reducer
) extends MasterWithRouter(xt, DFT[Double].|>, nPartitions)

object DFTMaster {
  type Reducer = (List[DblVec], String) => immutable.Seq[Double]
}

/**
  * '''Purpose''': Singleton to understand the behavior of Master-worker
  * design with Akka actors
  *
  * @author Patrick Nicolas
  * @note Scala for Machine Learning Chapter 16 Parallelism/Scala/Akka actors
  */
final class ActorManagerTest extends FlatSpec with Matchers with Logging {
  protected val name = "Actor manager"

  val NUM_WORKERS = 4
  val NUM_DATA_POINTS = 1000000

  // Synthetic generation function for multi-frequencies signals
  val h = (x: Double) => 2.0 * Math.cos(Math.PI * 0.005 * x) + // simulated first harmonic
    Math.cos(Math.PI * 0.05 * x) + // simulated second harmonic
    0.5 * Math.cos(Math.PI * 0.2 * x) + // simulated third harmonic
    0.2 * Random.nextDouble // noise

  type DblVec = Vector[Double]

  // User defined method that define the aggregation of the results
  // for the discrete Fourier transform for each worker actor
  def fReduce(aggrBuffer: List[DblVec], descriptor: String): immutable.Seq[Double] = {
    def display(x: Array[Double]): Unit = {
      import org.scalaml.plots.{Legend, LightPlotTheme, LinePlot}
      val labels = Legend(
        name,
        "Distributed DFT- Akka",
        s"DFT-Akka Frequencies distribution for $descriptor",
        "frequencies"
      )

      LinePlot.display(x.toVector, labels, new LightPlotTheme)
    }

    // Aggregate the results by transposing the observations
    // and sum the value for each dimension...
    val results = aggrBuffer.transpose.map(_.sum).toSeq
    show(s"DFT display ${results.size} frequencies")
    display(results.toArray)
    results
  }

  it should s"$name Master-Worker model for Akka actors with router" in {
    show("$name Master-Worker model for Akka actors with router")

    val actorSystem = ActorSystem("System")

    val xt = Vector.tabulate(NUM_DATA_POINTS)(h(_))
    actorSystem.actorOf(Props(new DFTMasterWithRouter(xt, NUM_WORKERS, fReduce)), "MasterWithRouter") ! Start(1)
  }

  it should s"$name Master-Worker model for Akka actors without router" in {
    show(s"$name Master-Worker model for Akka actors without router")

    val actorSystem = ActorSystem("System")

    val xt = Vector.tabulate(NUM_DATA_POINTS)(h(_))

    // The argument specifies if the group of worker actors is supervised
    // by a routing actor or not..
    actorSystem.actorOf(Props(new DFTMasterWithRouter(xt, NUM_WORKERS, fReduce)), "MasterWithRouter") ! Start(1)
  }
}

// ----------------------------------  EOF ------------------------