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
package org.scalaml.unsupervised.kmeans

import org.scalaml.{Logging, Resource}
import org.scalaml.Predef.DblVec
import org.scalaml.plots.Legend
import org.scalaml.unsupervised.env.TestEnv
import org.scalaml.unsupervised.kmeans.Cluster.DistanceFunc
import org.scalaml.workflow.data.DataSource
import org.scalaml.util.Assertable
import org.scalatest.{FlatSpec, Matchers}

import scala.language.postfixOps
import scala.util.Try



/**
  * Unit test for K-means
  */
final class KMeansTest extends FlatSpec with Matchers with Logging with Assertable with TestEnv with Resource {
  protected[this] val name = "K-Means"

  final val relPath = "unsupervised/kmeans/"

  type VSeriesSet = Array[Vector[Array[Double]]]

  final val MAX_ITERS = 250
  final val metric: DistanceFunc[Double] = Cluster.euclidDistance[Double]
  val NORMALIZE = true

  it should s"$name evaluation clustering symbols" in {
    import scala.language.postfixOps
    show("Evaluation clustering symbols")

    val input = Array[String]("2", "3", "4", "7", "9", "10", "13", "15")
    execute(input)
  }


  it should s"$name evaluation clustering symbols 2" in {
    import scala.language.postfixOps
    show(s"Evaluation clustering symbols")

    val input = Array[String]("2", "3", "4", "7", "9", "10", "13", "15", "18", "22")
    execute(input)
  }

  it should s"$name clustering synthetic data" in {
    import scala.util.Random._
    show("Clustering synthetic data")

    val SCALE = 100
    val K = 3

    /**
      * Random value generator
      */
    def fGen(id: Int): Double = SCALE * (id * 10.0 + nextDouble)

    /*
       * Features vector
       * @param id Identifier for the feature
       * @param a first feature
       * @param b second feature
       */
    case class Feature(id: Int, a: Double, b: Double) {
      val x = Array[Double](a, b)
    }

    val indexedFeatures = Vector.tabulate(120)(n => {
      val id = scala.util.Random.nextInt(K)
      (Feature(n, fGen(id + 1), fGen(id + 1)), id)
    })


      // Create an ordered list of K groups of features sorted by their id
    val expected = indexedFeatures
                .groupBy(_._2)
                .values.toList
                .sortWith(_(0)._2 < _(0)._2)
                .map(_.sortWith(_._1.id < _._1.id))

    val features = indexedFeatures.unzip._1
    val kmeans = KMeans[Double](KMeansConfig(K, MAX_ITERS), metric, features.map(_.x))

    kmeans.model match {
      case Some(m) =>
          // sort the model or list of clusters by the sum of features of their centroid
        val clusters: List[Cluster[Double]] = m
        val sorted = clusters.sortWith(_.center.sum < _.center.sum)

          // Retrieve the membership for each cluster
        val memberShip = sorted.map(_.getMembers)

          // Extract the id of expected features for each cluster
        val expectedId = expected.map(_.map(_._1.id))

        memberShip.zip(expectedId).foreach { case (c, e) => assertList(c, e.toList) }
        show(s"$name members\n${memberShip.map(_.mkString(", ")).mkString("\n")}")

      case None => error(s"$name Failed building a model")
    }
  }

  private def toString(groups: Iterable[Vector[Any]]): String = {
      groups.map(_.map(_.toString).mkString("\n")).mkString("\n-------\n")
  }


  private def execute(args: Array[String]): Unit = {
    val START_INDEX = 70
    val NUM_SAMPLES = 50
      // Incorrect format throws an exception that is caught by the eval.test handler

    val KValues = args.map(_.toInt)

      // Nested function to compute the density of K clusters generated from a
      // set of observations obs. The condition on the argument are caught
      // by the K-means constructor.

    def getDensity(K: Int, obs: Vector[Array[Double]]): DblVec = {
      val kmeans = KMeans[Double](KMeansConfig(K, MAX_ITERS), metric, obs)
      kmeans.density.getOrElse(Vector.empty[Double])
    }

      // Extract the price of the security using a data source

    def getPrices: Try[VSeriesSet] = Try {
      val path = getPath(relPath).getOrElse(".")
      symbolFiles(path).map(t => {
        val src = DataSource(t, path, NORMALIZE, 1)
     //   DataSource(t, path, NORMALIZE, 1)
        src.map(ds => {
          val extracted = ds.|>(extractor)

          extracted.getOrElse(Vector.empty[Array[Double]])
        }).getOrElse(Vector.empty[Array[Double]])
      })
    }

      // Extract a subset of observed prices

    def getPricesRange(prices: VSeriesSet) = prices.view.map(_.head.toArray).map(_.drop(START_INDEX).take(NUM_SAMPLES))

      // Basic computation pipeline
    (for {
      // Retrieve the stocks' prices
      prices <- getPrices

        // Retrieve the stocks' price variation ranges
      values <- Try { getPricesRange(prices) }

        // Compute the density of the clusters
      stdDev <- Try { KValues.map(getDensity(_, values.toVector)) }

        // Generates the partial function for this K-means transformation
      pfnKmeans <- Try {
         KMeans[Double](KMeansConfig(5, MAX_ITERS), metric, values.toVector) |>
      }

        // Generate the clusters if the partial function is defined.
      if pfnKmeans.isDefinedAt(values.head)
        predict <- pfnKmeans(values.head)
    } yield {

        // Display training profile
      profile(values.toVector)
      val results =
          s"""Daily price for ${prices.size} stocks
                       							| Clusters density\n${stdDev.mkString("\n"
          )}""".
            stripMargin
      show(results)
    }).getOrElse( error("failed to train K-means"))
  }


  private def profile(obs: Vector[Array[Double]]): Boolean = {
    val legend = Legend(
      "Density",
      "K-means clustering convergence", "Recursions", "Number of reasigned data points"
    )
    KMeans[Double](KMeansConfig(8, MAX_ITERS), metric, obs).display("Re-assigned", legend)
  }

  private def toString(clusters: List[Cluster[Double]]): String = {
    val path = getPath(relPath).getOrElse(",")

    clusters.zipWithIndex.map {
      case (c, n) =>
        val membership = c.getMembers.map(m =>
          symbolFiles(path)(m).substring(0, symbolFiles(path)(m).indexOf(".") - 1))
        s"\nCluster#$n => ${membership.mkString(",")}"
    }.mkString(" ")
  }
}

// ---------------------------- EOF ----------------------------------------------
