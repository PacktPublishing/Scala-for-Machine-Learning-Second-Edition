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
package org.scalaml.spark.extensibility

import org.apache.spark.ml.evaluation.Evaluator
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.{Estimator, PipelineStage}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{Dataset, SparkSession}

import scala.reflect.ClassTag

/**
  * Class that select the continuous probability distribution that fits the most accurately
  * a given data set. The fitness of a probability distribution is computed as the Kullback-
  * Leibler divergence between the distribution and the input dataset.
  *
  * @author Patrick Nicolas
  * @since 0.99.2
  * @param pdfs Sequence of continuous probability distributions defined by their name and
  *             probability density functions
  */
@throws(classOf[IllegalArgumentException])
private[spark] class KullbackLeibler(
    pdfs: Iterable[(String, Double => Double)],
    override val uid: String = "KullbackLeibler0.99.2"
)(implicit sparkSession: SparkSession) extends Evaluator {
  require(pdfs.nonEmpty, "Needs at least 1 probability distributions")

  private[this] val sparkContext = sparkSession.sparkContext

  val name: String = "KullbackLeibler"
  type DataIter = Iterator[(Double, Double)]
  type DataSeq = Seq[(Double, Double)]
  type BrdcastData = Iterable[(String, Double => Double)]

  final def kl(data: Dataset[(Double, Double)]): Seq[Double] = kl(data.rdd)

  /**
    * Select the distribution that fit the most accurately with input data
    * set xy. This implementation relies on broadcasting the sequence of continuous
    * probability distributions functions.
    *
    * @param fileName name of the file
    * @return name of the distribution
    */
  @throws(classOf[IllegalArgumentException])
  final def fittestDistribution(fileName: String): String = {
    require(fileName.nonEmpty, "Can't fit a distribution for undefined dataset")

    // Loads the RDD from input file
    val inputRdd = sparkContext.textFile(fileName).map(_.split(","))
      .map(arr => (arr.head.toDouble, arr.last.toDouble))
    kl(inputRdd).zip(pdfs).minBy(_._1)._2._1
  }

  @throws(classOf[IllegalStateException])
  private def kl(data: RDD[(Double, Double)]): Seq[Double] = {
    val pdfs_broadcast = sparkContext.broadcast[BrdcastData](pdfs)

    val kl = data.mapPartitions((it: DataIter) => {
      val Eps = 1e-20
      val LogEps = Math.log(Eps)

      pdfs_broadcast.value.map {
        case (key, pdf) =>
          def exec(xy: DataSeq, pdf: Double => Double): Double = {
            -xy./:(0.0) {
              case (s, (x, y)) => {
                val px = pdf(x)
                val z = if (Math.abs(y) < Eps) px / Eps else px / y
                s + px * (if (z < Eps) LogEps else Math.log(z))
              }
            }
          }
          (key, exec(it.toSeq, pdf))
      }.iterator
    }).collect

    if (kl.size == 0)
      throw new IllegalStateException("Kullback_Leibler is undefined")
    // Aggregate the KL values and extract the pdf that produces the smallest KL value
    (0 until kl.size by pdfs.size)
      ./:(Array.fill(pdfs.size)(0.0))((sum, n) => {
        (0 until pdfs.size).foreach(j => sum.update(j, kl(n + j)._2))
        sum
      }).map(_ / kl.length)
  }

  override def copy(extra: ParamMap): KullbackLeibler = defaultCopy(extra)

  override def evaluate(dataset: Dataset[_]): Double = {
    require(pdfs.size == 1, "Evaluate only one probability distribution")
    dataset match {
      case input: Dataset[(Double, Double)] @unchecked => kl(input.rdd).head
      case _ => Double.NaN
    }
  }
}

private[scalaml] object KullbackLeibler {
  /**
    * Factory for multiple probability distributions
    *
    * @param pdfs set of probability distributions density functions
    * @param sparkSession implicit reference to the session lifecycle
    * @return Instance of KullbackLeibler
    */
  def apply(
    pdfs: Iterable[(String, Double => Double)]
  )(implicit sparkSession: SparkSession): KullbackLeibler = new KullbackLeibler(pdfs)

  /**
    * Factory for multiple probability distributions
    *
    * @param pdf Probability distribution density function
    * @param sparkSession implicit reference to the session lifecycle
    * @return Instance of KullbackLeibler
    */
  def apply(
    pdfName: String,
    pdf: Double => Double
  )(implicit sparkSession: SparkSession): KullbackLeibler =
    apply(Iterable[(String, Double => Double)]((pdfName, pdf)))
}

// ------------------------------  EOF ---------------------------------------------
