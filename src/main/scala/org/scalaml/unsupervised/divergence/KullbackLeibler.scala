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
package org.scalaml.unsupervised.divergence

import org.scalaml.Predef.Context.ToDouble
import org.scalaml.util.MathUtils.Histogram

/**
  * Implementation of the Kullback-Leibler divergence
  * @author Patrick Nicolas
  * @version 0.99.2
  * @param p Distribution or dataset to evaluate against a target
  * @param q Target distribution or data set
  * @param normalized Flag that indicates that the dataset/distribution to evaluate and the target
  *                   distribution have to be normalized
  * @tparam T Type of the feature
  * @see Scala for Machine Learning - Chapter 5 - Dimension Reduction / Divergences
  */
@throws(classOf[IllegalArgumentException])
private[scalaml] class KullbackLeibler[@specialized(Double) T: ToDouble](
  p: Seq[T],
  q: Seq[T],
  normalized: Boolean = true
)(implicit ordering: Ordering[T]) extends Divergence {

  require(p.size > 0 && q.size > 0, "Cannot compute KL with undefined data sets")
  implicit def toDouble(t: T): Double = implicitly[ToDouble[T]].apply(t)

  private[this] val (min, max): (Double, Double) = (
    Math.min(p.min, q.min),
    Math.max(p.max, q.max)
  )

  private def frequencies(input: Seq[T], nSteps: Int): Array[Int] = {
    val histogram = new Histogram(min, max)
    histogram.frequencies(nSteps, input.map(toDouble(_)))
  }

  /**
    * Compute the asymmetric Kullback-Leibler divergence
    * @param nSteps Number of steps or frequency bins used to compute the KL divergence
    * @return Divergence value
    */
  @throws(classOf[IllegalArgumentException])
  override def divergence(nSteps: Int): Double = {
    import Math._
    require(nSteps > 2, s"KullbackLeibler divergence not available for nSteps $nSteps < 2")

    val freq1 = frequencies(p, nSteps)
    val freq2 = frequencies(q, nSteps)
      // If a count for a specific bucket is zero then do not count it
    val kl = freq1.zip(freq2)./:(0.0) {
      case (kl, (f1, f2)) => {
        val num = if (f2 == 0) 1 else f2
        val den = if(f1 == 0) 1 else f1
        // The KL relative entropy
        kl + num * log(num) / den
      }
    }

    if(normalized) kl/Math.max(p.size , q.size) else kl
  }
}

// -------------------------------  EOF ------------------------------------------------