package org.scalaml.workflow

import org.scalaml.Logging
import org.scalaml.core.Design.{ConfigDouble, ConfigInt}
import org.scalaml.core.ETransform
import org.scalaml.Predef._
import org.scalaml.stats.MinMax
import org.scalatest.{FlatSpec, Matchers}

import scala.util.{Failure, Random, Success, Try}

final class WorkflowTest extends FlatSpec with Matchers with Logging {
  protected[this] val name = "Workflow for data pipeline"

  it should s"$name Illustration of a monadic workflow" in {

    val samples: Int = 100
    val normRatio = 10
    val splits = 4

    val g = (x: Double) => Math.log(x + 1.0) + Random.nextDouble

    val workflow = new Workflow[Double => Double, DblVec, DblVec, Int] with Sampling[Double => Double, DblVec] with Normalization[DblVec, DblVec] with Aggregation[DblVec, Int] {

      val sampler = new ETransform[Double => Double, DblVec](ConfigInt(samples)) {

        override def |> : PartialFunction[Double => Double, Try[DblVec]] = {
          case f: (Double => Double) => Try {
            val sampled: DblVec = Vector.tabulate(samples)(n => f(n.toDouble / samples))
            show(s"$name sampling : ${sampled.mkString(",")}")
            sampled
          }
        }
      }

      val normalizer = new ETransform[DblVec, DblVec](ConfigDouble(normRatio)) {

        override def |> : PartialFunction[DblVec, Try[DblVec]] = {
          case x: DblVec if x.nonEmpty => Try {
            val minMax = MinMax[Double](x).map(_.normalize(0.0, 1.0)).getOrElse(Vector.empty[Double])
            show(s"$name normalization : ${minMax.mkString(",")}")
            minMax
          }
        }
      }

      val aggregator = new ETransform[DblVec, Int](ConfigInt(splits)) {

        override def |> : PartialFunction[DblVec, Try[Int]] = {
          case x: DblVec if x.nonEmpty => Try {
            show(s"$name aggregation")
            Range(0, x.size).find(x(_) == 1.0).getOrElse(-1)
          }
        }
      }
    }
    (workflow |> g) match {
      case Success(res) => show(s"$name result = ${res.toString}")
      case Failure(e) => error(s"$name", e)
    }
  }
}


// ---------------------------------------  EOF ----------------------------------------------
