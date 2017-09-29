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
package org.scalaml.unsupervised.functionapprox

import org.scalaml.Logging
import org.scalaml.Predef.Context.ToDouble
import org.scalatest.{FlatSpec, Matchers}

import scala.util.Random

/**
 * Unit test for Function approximation
 */
final class FunctionApproxTest extends FlatSpec with Matchers with Logging {
  protected[this] val name = "Function Approximation"
    // Simplest data point definition
  case class DataPoint( id: String, value: Double )
  final val expected = Math.log( _ )

  it should s"$name using a non-resizable histogram" in {
    show(s"$name using a non-resizable histogram")

    implicit val dataPoint2Double = new ToDouble[DataPoint] {
      def apply( dataPoint: DataPoint ): Double = dataPoint.value
    }

    val input = Array.tabulate( 10000 )( n => {
      val x = 1.0 + 9.0 * Random.nextDouble
      ( DataPoint( n.toString, x ), expected( x ) )
    } )

    val testSample = List[DataPoint](
      DataPoint( "2001", 2.8 ),
      DataPoint( "2002", 5.5 ),
      DataPoint( "2003", 7.1 )
    )

    val error2 = error( new HistogramApprox[DataPoint]( 2, input ), testSample )
    show( s"$name error 2 $error2" )

    val error5 = error( new HistogramApprox[DataPoint]( 5, input ), testSample )
    show( s"$name error 5 $error5" )

    val error10 = error( new HistogramApprox[DataPoint]( 10, input ), testSample )
    show( s"$name error 10 $error10" )

    val error25 = error( new HistogramApprox[DataPoint]( 25, input ), testSample )
    show( s"$name error 25 $error25" )

    val error100 = error( new HistogramApprox[DataPoint]( 100, input ), testSample )
    show( s"$name error 100 $error100" )
  }

  private def error( functionApprox: FunctionApprox[DataPoint], testSample: List[DataPoint] ): Double =
    Math.sqrt( testSample./:( 0.0 )( ( s, dataPoint ) => {
      val delta = functionApprox.predict( dataPoint ) - expected( dataPoint.value )
      s + delta * delta
    } ) )
}

// -----------------------  EOF ----------------------------------------------------