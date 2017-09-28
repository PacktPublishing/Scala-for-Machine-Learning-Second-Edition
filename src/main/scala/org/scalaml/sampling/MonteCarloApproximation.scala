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
package org.scalaml.sampling

import scala.util.Random

/**
  * Class that implement the Monte Carlo integration for a given single variable function
  * f and a number of points
  * @author Patrick Nicolas
  * @version 0.99.2
  *
  * @param f The single variable function to integrate
  * @param numPoints The total number of points used in the integration (or summation)
  *
  * @see Scala for Machine Learning Chapter 8, Monte Carlo Inference - Monte Carlo approximation.
  */
private[scalaml] class MonteCarloApproximation(f: Double => Double, numPoints: Int) {

  /**
    * Method that compute the integral sum (under the curve defined by the function f)
    * @param from Starting value of integration interval
    * @param to End value of the integration interval
    * @return Integral sum
    */
    def sum(from: Double, to: Double): Double = {
        // Get the minimum and maximum values for the function
      val (min, max) = getBounds(from, to)
      val width = to - from
      val height = if (min >= 0.0) max else max - min
        // compute the enclosing area (rectangle)
      val outerArea = width * height
      val randomx = new Random(System.currentTimeMillis)
      val randomy = new Random(System.currentTimeMillis + 42L)

        // Monte Carlo simulator for the  function
      def randomSquare: Double = {
        val numInsideArea = Range(0, numPoints)./:(0)(
          (s, n) => {
            val ptx = randomx.nextDouble * width + from
            val pty = randomy.nextDouble * height
              // update the seeds
            randomx.setSeed(randomy.nextLong)
            randomy.setSeed(randomx.nextLong)

            s + (if (pty > 0.0 && pty < f(ptx)) 1
            else if (pty < 0.0 && pty > f(ptx)) -1
            else 0)
          }
        )
        numInsideArea.toDouble * outerArea / numPoints
      }
      randomSquare
    }

    // Compute the bounds for the y values of the function
  private def getBounds(from: Double, to: Double): (Double, Double) = {
    def updateBounds(y: Double, minMax: (Double,Double)): Int = {
      var flag = 0x00
      if (y < minMax._1) flag += 0x01
      if (y > minMax._2) flag += 0x02
      flag
    }
      // extract the properties for the integration step
    val numSteps = Math.sqrt(numPoints).floor.toInt
    val stepSize = (to - from) / numSteps

    (0 to numSteps)./:((Double.MaxValue, -Double.MaxValue))(
      (minMax, n) => {
        val y = f(n * stepSize + from)
        updateBounds(y, minMax) match {
          case 0x01 => (y, minMax._2)
          case 0x02 => (minMax._1, y)
          case 0x03 => (y, y)
          case _ => minMax
        }
      }
    )
  }
}

// --------------------------  EOF ------------------------------------------
