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
package org.scalaml.validation

import org.scalaml.Logging
import org.scalaml.Predef._
import org.scalatest.{FlatSpec, Matchers}


/**
  * Test to compute some basic quality metrics for binary classifiers
  */
final class BinaryValidationTest extends FlatSpec with Matchers with Logging {
  protected[this] val name = "Binary classifier validation"

  final val SOURCE = Vector[Int]( 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0 )
  final val EXPECTED = Vector[Int]( 1, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0 )
  final val PREDICTOR = ( x: Features ) => x( 0 ).floor.toInt

  final val PRECISION = 0.78
  final val RECALL = 1.0
  final val F1 = 0.875

  it should s"$name scoring" in {
    show(s"$name scoring")
    // Display the profile of different Fn score
    displayFnScore()

    // Create a stats object and compute the normalized and zScored value
    val predicted = SOURCE.map( _.toDouble ).map( Array[Double]( _ ) )
    val validator = new BinaryValidation( EXPECTED, predicted )( PREDICTOR )

    validator.precision should be( PRECISION +- 1e-2 )
    show( s"precision = ${validator.precision}" )

    validator.recall should be( RECALL +- 1e-2 )
    show( s"recall = ${validator.recall}" )

    validator.f1 should be( F1 +- 1e-2 )
    show( s"F1 = ${validator.f1}" )
  }

  private def displayFnScore(): Unit = {
    import org.scalaml.plots._
    val info = Legend(
      "ValidationEval",
      "Validation: F1, F2 and F3 score with recall = 0.5",
      "Precision",
      "F values"
    )

    val R = 0.5
    val f1Precision = ( p: Double ) => 2.0 * p * R / ( p + R )
    val f2Precision = ( p: Double ) => 5 * p * R / ( 4 * p + R )
    val f3Precision = ( p: Double ) => 10 * p * R / ( 9 * p + R )

    val entries = List[( DblVec, String )](
      ( Vector.tabulate( 98 )( n => f1Precision( 0.01 * ( n + 1 ) ) ), "F1 (precision)" ),
      ( Vector.tabulate( 98 )( n => f2Precision( 0.01 * ( n + 1 ) ) ), "F2 (precision)" ),
      ( Vector.tabulate( 98 )( n => f3Precision( 0.01 * ( n + 1 ) ) ), "F3 (precision)" )
    )
    LinePlot.display( entries, info, new LightPlotTheme )
  }
}


// -------------------------  EOF -------------------------------------