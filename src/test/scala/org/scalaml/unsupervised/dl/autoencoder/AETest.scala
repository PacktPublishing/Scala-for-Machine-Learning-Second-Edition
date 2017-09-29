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
package org.scalaml.unsupervised.dl.autoencoder

import org.scalaml.{Logging, Resource}
import org.scalaml.Predef.DblVec
import org.scalaml.trading.GoogleFinancials.close
import org.scalaml.workflow.data.DataSource
import org.scalatest.{FlatSpec, Matchers}


final class AETest extends FlatSpec with Matchers with Logging with Resource {
  protected val name: String = "Auto-Encoder"

  it should s"$name single hidden layer" in {
    show( "Single hidden layer")

    val REL_PATH = "unsupervised/ae/"
    val ALPHA = 0.8
    val ETA = 0.05
    val NUM_EPOCHS = 2500
    val EPS = 1e-6
    val THRESHOLD = 0.25
    val LAMBDA = 0.18
    val BETA = 0.3

    val symbols = Array[String](
      "FXE", "FXA", "SPY", "GLD", "FXB", "FXF", "FXC", "FXY", "CYB"
    )

    val STUDIES = List[Array[String]](
      Array[String]("FXY", "FXC", "GLD", "FXA"),
      Array[String]("FXE", "FXF", "FXB", "CYB"),
      Array[String]("FXE", "FXC", "GLD", "FXA", "FXY", "FXB"),
      Array[String]("FXC", "FXY", "FXA"),
      Array[String]("CYB", "GLD", "FXY"),
      symbols
    )

    def index: Map[String, Int] =  {
      import scala.collection.mutable.HashMap
      symbols.zipWithIndex./:(HashMap[String, Int]())((mp, si)  => mp += ((si._1, si._2))).toMap
    }

    val path: String = getPath(REL_PATH).getOrElse(".")
    val prices = symbols.map(s => DataSource(s"$path$s.csv", true, true, 1))
      .map( _.flatMap(_.get(close))).filter(_.isSuccess).map(_.get)

    val config = AEConfig(ALPHA, ETA, LAMBDA, BETA, NUM_EPOCHS, EPS)
    val obs = symbols.flatMap( index.get(_)).map(prices(_).toArray)

    val xv = obs.tail.transpose.dropRight(1)

    val ae = AE(config, 8, xv.toVector)

    ae.model match {
      case Some(aeModel) =>
        if(aeModel.synapses.nonEmpty) {
          val inputSynapse = aeModel.synapses.head
          show(s"$name output synapse(0)(0) ${inputSynapse(0)(0)}")
          show(s"$name output synapse(0)(1) ${inputSynapse(0)(1)}")
          show(s"$name output synapse(1)(0) ${inputSynapse(1)(0)}")
          show(s"$name output synapse(1)(1) ${inputSynapse(1)(1)}")
        }
        else
          fail(s"$name Model weights with improper size")
      case None => fail(s"$name could not generate a model")
    }
  }
}

// ---------------------------------  EOF ----------------------------------------------------------------------------