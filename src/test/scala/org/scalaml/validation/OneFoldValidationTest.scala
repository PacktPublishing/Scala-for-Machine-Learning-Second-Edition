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
import org.scalaml.Predef.Features
import org.scalatest.{FlatSpec, Matchers}

final class OneFoldValidationTest extends FlatSpec with Matchers with Logging {
  protected[this] val name = "One-Fold Validation"

  // A sample of observed features
  final val INPUT = Vector[Features](
    Array[Double]( 3.5, 1.6 ),
    Array[Double]( 4.5, 2.6 ),
    Array[Double]( 5.5, 3.6 ),
    Array[Double]( 6.5, 4.6 ),
    Array[Double]( 7.5, 5.6 ),
    Array[Double]( 8.5, 6.6 ),
    Array[Double]( 9.5, 7.6 ),
    Array[Double]( 10.5, 8.6 ),
    Array[Double]( 11.5, 9.6 ),
    Array[Double]( 12.5, 10.6 ),
    Array[Double]( 13.5, 11.6 ),
    Array[Double]( 14.5, 12.6 ),
    Array[Double]( 15.5, 13.6 ),
    Array[Double]( 16.5, 14.6 ),
    Array[Double]( 17.5, 15.6 ),
    Array[Double]( 18.5, 16.6 ),
    Array[Double]( 19.5, 17.6 )
  )

  it should s"$name validation random scenario" in {
    show(s"$name validation random scenario")
    // sample of labels
    val expected = Vector[Int]( 0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1 )

    oneFoldValidate(expected)
  }

  it should s"$name validation zero scenario" in {
    show(s"$name validation zero scenario")

    // sample of labels
    val expected = Vector[Int]( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    oneFoldValidate(expected)
  }

  it should s"$name validation all 1 scenario" in {
    show("$name validation all 1 scenario")
    // sample of labels
    val expected = Vector[Int]( 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 )
    oneFoldValidate(expected)
  }

  private def oneFoldValidate(expected: Vector[Int]): Unit = {
    if ( INPUT.size == expected.size )
      show( s"$name Incorrect size" )

    // Create a 1 fold validation
    val xValidation = new OneFoldValidation[Double]( INPUT, expected, 0.6 )

    // extracts the training and validation sets
    val trainSet = xValidation.trainingSet.map {
      case ( x, n ) => s"${x.mkString( "," )} | $n"
    }
    val valSet = xValidation.validationSet.map {
      case ( x, n ) => s"${x.mkString( "," )} | $n"
    }
    // Validate the size of training and validation set
    if ( trainSet.size + valSet.size == INPUT.size )
      show( s"$name Incorrect size" )
    show( s"$name Training set:\n${trainSet.mkString( "\n" )}" )
    show( s"$name Validation set:\n${valSet.mkString( "\n" )}" )
  }
}


// ------------------------------  EOF -----------------------------------------------