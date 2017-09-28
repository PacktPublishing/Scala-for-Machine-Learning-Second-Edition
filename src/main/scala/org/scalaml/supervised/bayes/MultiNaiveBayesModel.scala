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
package org.scalaml.supervised.bayes

import org.scalaml.Predef.Context._
import NaiveBayesModel._

/**
 * Defines a Multi-class (or multinomial) Naive Bayes model for n classes.
 * The number of classes is defined as likelihoodSet.size. The binomial Naive Bayes model,
 * BinNaiveBayesModel, should be used for the two class problem.
 * @tparam T type of features in each observation
 * @constructor Instantiates a multi-nomial Naive Bayes model (number classes > 2)
 * @throws IllegalArgumentException if any of the class parameters is undefined
 * @param likelihoodSet  List of likelihood or priors for every classes in the model.
 * @author Patrick Nicolas
 * @since 0.98 February 11, 2014
 * @version 0.99.2
 * @see Scala for Machine Learning  Chapter 6 "Naive Bayes Models" / Naive Bayes Classifiers
 */
@throws(classOf[IllegalArgumentException])
private[scalaml] class MultiNaiveBayesModel[T] protected (
    likelihoodSet: Seq[Likelihood[T]]
) extends NaiveBayesModel[T] {

  require(
    likelihoodSet.nonEmpty,
    "MultiNaiveBayesModel Cannot classify using Multi-NB with undefined classes"
  )

  /**
   * Classify a new observation (or vector) using the Mult-inomial Naive Bayes model.
   * @param x new observation
   * @return the class ID the new observations has been classified.
   * @throws IllegalArgumentException if any of the observation is undefined.
   */
  override def classify(x: Array[T]): Int = {
    require(x.length > 0, "MultiNaiveBayesModel.classify Vector input is undefined")

    // The classification is performed by ordering the class according to the
    // log of their posterior probability and selecting the top one (highest
    // posterior probability)

    val <<< = (p1: Likelihood[T], p2: Likelihood[T]) => p1.score(x) > p1.score(x)
    likelihoodSet.sortWith(<<<).head.label
  }

  override def likelihoods: Seq[Likelihood[T]] = likelihoodSet

  override def toString(labels: Array[String]): String = {
    require(labels.length > 0, "MultiNaiveBayesModel.toString Vector input is undefined")

    likelihoodSet.zipWithIndex
      .map { case (lp, n) => s"\nclass$n : ${lp.toString(labels)}" }
      .mkString(",")
  }

  override def toString: String = likelihoodSet.mkString("\n")
}

/**
 * Companion object for the multinomial Naive Bayes Model. The singleton
 * is used to define the constructor of MultiNaiveBayesModel
 *
 * @author Patrick Nicolas
 * @since 0.98 February 10, 2014
 * @version 0.99.2
 * @see Scala for Machine Learning  Chapter 5 "Naive Bayes Models" / Naive Bayes Classifiers
 */
private[scalaml] object MultiNaiveBayesModel {
  /**
   * Default constructor for the multinomial Naive Bayes model as instance of
   * MultiNaiveBayesModel
   * @tparam T type of features in each observation
   * @param likelihoodSet  List of likelihood or priors for every classes in the model.
   */
  def apply[T](likelihoodSet: Seq[Likelihood[T]]): MultiNaiveBayesModel[T] =
    new MultiNaiveBayesModel[T](likelihoodSet)
}

// --------------------------------  EOF --------------------------------------------------------------