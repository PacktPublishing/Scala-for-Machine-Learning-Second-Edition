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
package org.scalaml.ga

import scala.collection.mutable.ArrayBuffer
import org.scalaml.{Logging, Resource}
import org.scalaml.Predef.{DblPair, DblVec}
import org.scalaml.trading.Signal
import org.scalatest.{FlatSpec, Matchers}
import Gene._
import org.scalaml.ga.Chromosome.Pool
import org.scalaml.trading.operator.{GREATER_THAN, LESS_THAN}
import org.scalaml.util.FormatUtils._
import org.scalaml.workflow.data.DataSource
import org.scalaml.trading.YahooFinancials.{volatility, volume, _}
import org.scalaml.trading.StrategyFactory
import org.scalaml.stats.TSeries._

import scala.util.{Failure, Random, Success, Try}

/**
  * Unit test for the generic algorithm
  */
final class GATest extends FlatSpec with Matchers with Logging with Resource {
  protected val name: String = "Genetic algorithm"

  it should s"$name evaluation" in {
    import scala.language.postfixOps
    show(s"$name evaluation")

    val Xover = 0.8					      // Probability or ratio for cross-over
    val Mu = 0.4					        // Probability or ratio for mutation
    val MaxCycles = 1000			    // Maximum number of iterations during the optimization
    val CutoffSlope = -0.001		  // Slope for the linear soft limit
    val CutoffIntercept = 1.003	  // Intercept value for the linear soft limit
    val RCoef: Int = 1024         // Quantization ratio for conversion Int <-> Double

    val softLimit = (n: Int) => CutoffSlope*n + CutoffIntercept
      // Default data conversion
    implicit val quant = Quantization[Double]((x: Double) => (x * RCoef).floor.toInt, (n: Int) => n.toDouble/RCoef)
    implicit val encoding = defaultEncoding


    val relative = (xy: DblPair) => xy._2/xy._1 -1.0
    val invRelative = (xy: DblPair) => xy._2/xy._1 -1.0
      /**
        * Define the scoring function for the chromosomes (i.e. Trading strategies)
        * as the sum of the score of the genes (i.e. trading signals) in this chromosome (i.e strategy).
        */
    val scoring = (chr: Chromosome[Double, Signal]) =>  {
      val signals: List[Gene[Double]] = chr.code
      chr.cost = Math.log(signals.map(_.score).sum + 1.0)
    }


    def delta(series: DblVec, a: Double): Try[DblVec] = Try(zipWithShift(series, 1).map{ case(x, y) => a*(y/x - 1.0)})
    /*
    * Create Trading strategies by loading price data from Yahoo financial tables.
    */
    import Chromosome._
    val resPath = "ga/GS.csv"
    def createStrategies: Try[Pool[Double, Signal]] = for {
        path <- getPath(resPath)
        src <- DataSource(path, false, true, 1)
        price <- src.get(adjClose)
        dPrice <- delta(price, -1.0)
        volume <- src.get(volume)
        dVolume <- delta(volume, 1.0)
        volatility <- src.get(volatility)
        dVolatility <- delta(volatility, 1.0)
        vPrice <- src.get(vPrice)
      } yield {
        show(s"""$name GS Stock price variation
                				| ${format(dPrice, "GS stock price daily variation", SHORT)}""".stripMargin )

        val NumSignalsPerStrategy=3 // Number of trading signals per trading strategy
        // (= number of genes in a chromosome)
        val factory = StrategyFactory(NumSignalsPerStrategy)

        val avWeights = dPrice.sum/dPrice.size
        val weights = Vector.fill(dPrice.size)(avWeights)
        factory +=  ("dVolume", 1.1, GREATER_THAN, dVolume, weights)
        factory +=  ("volatility", 1.3, GREATER_THAN, volatility.drop(1), weights)
        factory +=  ("change", 0.8, LESS_THAN, vPrice.drop(1), weights)
        factory +=  ("dVolatility", 0.9, GREATER_THAN, dVolatility, weights)

        factory.strategies
      }


    /**
        * Monitoring function for the GA execution
        */
    val averageCost = ArrayBuffer[Double]()
    val tracker = (p: Population[Double, Signal]) => averageCost.append(p.averageCost)


      // Create trading strategies
    createStrategies.map(strategies => {
        show(s"\n${strategies.mkString("\n")}")

        // Initialize the population with a upper bound of 16 times
        // the initial number of strategies
        val initial = Population[Double, Signal](strategies.size << 4, strategies)

        show(s"$name ${initial.symbolic}")

        // Configure, instantiates the GA solver for trading signals
        val config = GAConfig(Xover, Mu, MaxCycles, softLimit)
        val solver = GASolver[Double, Signal](config, scoring, Some(tracker))


        // Extract the best population and the fittest chromosomes = trading strategies
        // from this final population.
        (solver |> initial).map(_.fittest.map(_.symbolic).getOrElse("NA")) match {
          case Success(results) =>
            display(averageCost.toArray)
            show(results)

          case Failure(e) => error(s"$name training ", e)
        }
      }).getOrElse(error(s"$name test failed"))

    def display(data: Array[Double]): Unit = {
      import org.scalaml.plots.{Legend, LinePlot, LightPlotTheme}

      val labels = new Legend(name, "Genetic algorithm convergence", "Recursions", "Average cost")
      LinePlot.display(data, labels, new LightPlotTheme)
    }
  }

  it should s"$name for function maximization" in {
    /**
      * Class that models a simple gradient of a function. This simplistic representation of
      * the gradient is defined a gene with the delta_x value as target and FDirector of change
      * as operator (type FDirection). The initial value x0 is also a class parameter.
      * {{{
      *   f(x0 + delta) = f(x0) + gradient*delta
      *   gradient = [f(x0 + delta) - f(x0)]/delta
      * }}}
      *
      * @param id    Identifier of this gradient
      * @param delta delta(x) = x - x0 used in the gradient denominator
      * @param op    Direction of the change as incr for delta > 0 and decr for delta < 0
      * @param x0    Initial value of the function.
      */
    class FGradient(
      id: String,
      delta: Double,
      op: FDirection,
      x0: Double)(implicit quant: Quantization[Double], geneBits: Encoding) extends Gene[Double](id, delta, op) {

      /**
        * Virtual constructor used in cloning, mutation and cross-over of gene, that
        * generate an instance of appropriate type.
        *
        * @param id    identifier for the gradient
        * @param delta delta value used in the gradient
        * @param op    simple operator that increase or decrease the value x for the operator
        * @return a new instance of the gradient with delta and direction modified through
        *         genetic reproduction. The new gradient has the same original value x0 as its parent.
        */
      override def toGene(id: String, delta: Double, op: Operator): Gene[Double] =
        new FGradient(id, delta, op.asInstanceOf[FDirection], x0)

      /**
        * Action of the gradient f(x + delta) = f(x) + gradient
        *
        * @return New value of the function
        */
      override def score: Double = op(x0, delta)

      /**
        * Symbolic representation of the gradient
        */
      override def symbolic: String = s"$id: ${op.toString}$delta"
    }

    /**
      * Class that defines the direction of a change (delta) to compute the gradient of a
      * function.
      *
      * @param _type Identifier for the directional operator
      * @param f     Function that compute the value f(x0 + delta)
      */
    class FDirection(_type: Int, val f: (Double, Double) => Double) extends Operator {
      /**
        * Type of the directional operator (0 for delta > 0, 1 for delta < 0)
        */
      override def id: Int = _type

      /**
        * Return the actual type of the operator, FIncrease or FDecrease
        *
        * @param _type of the directional operator
        * @return instance of the directional operator (singleton)
        */
      override def apply(_type: Int): FDirection = if (_type == 0) FIncrease else FDecrease
      def apply(x: Double, incr: Double): Double = f(x, incr)

      override def toString: String = s"$name Error for ${_type}"
    }

    /**
      * Singleton that defines the increase operator
      */
    object FIncrease extends FDirection(0, (x: Double, incr: Double) => x + incr) {
      override def toString: String = "incr "
    }

    /**
      * Singleton that defines the decrease operator
      */
    object FDecrease extends FDirection(1, (x: Double, incr: Double) => x - incr) {
      override def toString: String = "decr "
    }

    type ChAction = Chromosome[Double, FGradient]

    val Xover = 0.9 // Probability or ratio for cross-over
    val Mu = 0.87 // Probability or ratio for mutation
    val MaxCycles = 200 // Maximum number of iterations during the optimization
    val softLimit = (n: Int) => 0.95
    val SEED_SIZE = 10
    val SEED_SIZE_2 = SEED_SIZE << 1

    // Minimum values is (x = 4.0, y = 2.5)
    val gf1 = (x: Double) => {
      val y = x - 4.0; y * y + 2.5
    }

    // Chromosome scoring function. The scoring function adds the delta from all the
    // Gradients (genes), then apply the function to the delta as
    // f(x) = f(x0) + gradient*(delta1 + delta2 + ..)
    val scoring = (chr: Chromosome[Double, FGradient]) => {
      val fActions: List[Gene[Double]] = chr.code
      // summation of all the delta values
      val sumDelta = fActions.map(_.score).reduce(_ + _)
      chr.cost = gf1(sumDelta)
    }
    /*
       * Discretize the 32-bit value into R = 1024 levels
       */
    val RCoef = 1024
    val EncodingSize = 32
    implicit val quantization = Quantization[Double]((x: Double) => (x * RCoef).floor.toInt, (n: Int) => n.toDouble/RCoef)
    implicit val geneBits = new Gene.Encoding(EncodingSize, 1)


      // Create the initial set or population of FAction
    def createInitialPop(x0: Double): Pool[Double, FGradient] = {
      val fActionList = List.tabulate(SEED_SIZE_2)(n =>
        new FGradient(n.toString,
          (SEED_SIZE_2 << 2) * Random.nextDouble,
          if (Random.nextInt(2) == 0x01) FIncrease else FDecrease,
          x0)
      )

      val fActionList_1: List[FGradient] = fActionList.take(SEED_SIZE)
      val fActionList_2: List[FGradient] = fActionList.takeRight(SEED_SIZE)

      (0 until 30)./:(ArrayBuffer[ChAction]())((buf, n) => {
        val xs = List[FGradient](
          fActionList_1(Random.nextInt(SEED_SIZE)),
          fActionList_2(Random.nextInt(SEED_SIZE))
        )
        buf += new Chromosome[Double, FGradient](xs)
      })
    }

    // Initialize the encoding of the gradient as a 32 delta value and a single bit directional
    // operator 0 for increase, 1 for decrease
    val x0 = 8
    val MaxPopulationSize = 120
    // Initialize the population and the maximum number of chromosomes or
    // solution candidates
    val population = Population[Double, FGradient](MaxPopulationSize, createInitialPop(x0))
    val config = GAConfig(Xover, Mu, MaxCycles, softLimit)

    // Define a monitoring callback to trace each reproduction cycle
    val monitor: Option[Population[Double, FGradient] => Unit] = Some(
      (current: Population[Double, FGradient]) => {
        val topChromosomes = current.fittest(5).map(_.toArray).getOrElse(Array.empty)
        if (!topChromosomes.isEmpty) {
          topChromosomes.foreach(ch => show(ch.symbolic))
          show(s"$name average cost: ${current.averageCost}")
        }
      }
    )

    // Instantiate and execute the genetic solver
    val pfnSolver = GASolver[Double, FGradient](config, scoring, monitor) |>

    val result = pfnSolver(population).map(best => {
      val fittest = best.fittest.get
      scoring(fittest)
      fittest.symbolic
    })
    show(s"$name solution $result")
  }
}


// ------------------------------------  EOF ------------------------------------------------------------
