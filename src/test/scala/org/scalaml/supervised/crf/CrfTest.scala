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
package org.scalaml.supervised.crf

import org.scalaml.{Logging, Resource}
import org.scalaml.libraries.crf.CrfAdapter.CrfSeqDelimiter
import org.scalaml.util.FormatUtils._
import org.scalatest.{FlatSpec, Matchers}


final class CrfTest  extends FlatSpec with Matchers with Logging with Resource {
  protected[this] val name: String = "Conditional Random Fields"

  private val LAMBDA = 0.5
  private val NLABELS = 9
  private val MAX_ITERS = 100
  private val W0 = 0.7
  private val EPS = 1e-3

  it should s"$name entries 1" in {
    show("Evaluate Conditional Random Fields entries 1")

    val PATH = "supervised/crf/rating"

    val config = CrfConfig(W0 , MAX_ITERS, LAMBDA, EPS)
    val delimiters = new CrfSeqDelimiter(",\t/ -():.;'?#`&_", "//", "\n")

    // Step 2: Create a CRF model (weights) through training by instantiating
    // 				 the Crf class
    val path = getPath(PATH).getOrElse(".")
    val crf = Crf(NLABELS, config, delimiters, path)

    // Step 3: Display the model for the Crf classifier
    if(crf.weights.isDefined ) {
      val results = crf.weights.map( w => {
        display(w)
        format(w, "CRF weights", SHORT)
      })
      show(s"$name weights for conditional random fields\n${results.mkString(",")}")
    }
    else
      error(s"$name Failed to train the conditional random field")
  }


  it should s"$name evaluate entries 2" in {
    show("Evaluate CRF 2")

    val PATH = "supervised/crf/rating2"

    val config = CrfConfig(W0 , MAX_ITERS, LAMBDA, EPS)
    val delimiters = new CrfSeqDelimiter(",\t/ -():.;'?#`&_", "//", "\n")

    // Step 2: Create a CRF model (weights) through training by instantiating
    // 				 the Crf class
    val path = getPath(PATH).getOrElse(".")
    val crf = Crf(NLABELS, config, delimiters, path)

    // Step 3: Display the model for the Crf classifier
    if(crf.weights.isDefined ) {
      val results = crf.weights.map( w => {
        display(w)
        format(w, "CRF weights", SHORT)
      })
      show(s"$name weights for conditional random fields\n${results.mkString(",")}")
    }
    else
      error(s"$name Failed to train the conditional random field")
  }

  /**
    * Display the model parameters (or weights) on a line plot
    */
  private def display(w: Array[Double]): Unit = {
    import org.scalaml.plots.{Legend, LightPlotTheme, LinePlot}

    val labels = Legend(
      name, "Conditional random fields weights vs lambda", "Lambda distribution", "weights"
    )
    LinePlot.display(w, labels, new LightPlotTheme)
  }
}
