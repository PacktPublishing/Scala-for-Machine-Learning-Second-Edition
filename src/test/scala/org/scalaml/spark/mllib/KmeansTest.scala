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

import org.apache.log4j.{Level, Logger}
import org.apache.spark.storage.StorageLevel
import org.apache.spark.{SparkConf, SparkContext}
import org.scalaml.{Logging, Resource}
import org.scalaml.Predef._
import org.scalaml.stats.TSeries._
import org.scalaml.trading.YahooFinancials
import org.scalaml.workflow.data.DataSource
import org.scalatest.FunSuite
import org.scalatest.concurrent.ScalaFutures

import scala.concurrent.Future


final class KmeansTest extends FunSuite with ScalaFutures with Logging with Resource {
  import scala.concurrent.ExecutionContext.Implicits.global

  protected[this] val name = "Spark MLlib K-Means"
  private val K = 8
  private val NRUNS = 4
  private val MAXITERS = 60
  private val PATH = "spark/CSCO.csv"
  private val CACHE = false

  test(s"$name evaluation") {
    show(s"Evaluation")

    Logger.getRootLogger.setLevel(Level.ERROR)
    // The Spark configuration has to be customize to your environment
    val sparkConf = new SparkConf().setMaster("local")
      .setAppName("Kmeans")
      .set("spark.executor.memory", "4096m")

    implicit val sc = SparkContext.getOrCreate(sparkConf) // no need to load additional jar file

    val kmeanClustering: Option[Kmeans] = extract.map(input => {
      val volatilityVol = zipToSeries(input._1, input._2).take(500)

      val config = new KmeansConfig(K, MAXITERS, NRUNS)
      val rddConfig = RDDConfig(CACHE, StorageLevel.MEMORY_ONLY)
      Kmeans(config, rddConfig, volatilityVol)
    })

      // Wraps into a future to enforce time out in case of a straggler
    val ft = Future[Boolean] { predict(kmeanClustering) }
    whenReady(ft) { result => assert(result) }
    sc.stop
  }

   private def predict(kmeanClustering: Option[Kmeans]): Boolean = {
     kmeanClustering.map(kmeansCluster => {
       val obs = Array[Double](0.1, 0.9)
       val clusterId1 = kmeansCluster |> obs
       show(s"(${obs(0)},${obs(1)}) => Cluster #$clusterId1")

       val obs2 = Array[Double](0.56, 0.11)
       val clusterId2 = kmeansCluster |> obs2
       val result = s"(${obs2(0)},${obs2(1)}) => Cluster #$clusterId2"
       show(s"$name result: $result")
     })
     true
   }

  private def extract: Option[(DblVec, DblVec)] = {
    import scala.util._
    val extractors = List[Array[String] => Double](
      YahooFinancials.volatility,
      YahooFinancials.volume
    )

    DataSource(getPath(PATH).get, true).map(_.|>) match {
      case Success(pfnSrc) => pfnSrc(extractors).map(res => ((res(0).toVector, res(1).toVector))).toOption
      case Failure(e) =>
        failureHandler(e)
        None
    }
  }
}


// ---------------------------------  EOF -------------------------------------------------