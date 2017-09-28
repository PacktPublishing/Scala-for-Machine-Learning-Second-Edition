package org.scalaml

import scala.util.Try

/**
 * Package that encapsulates the types and conversion used in the Scala for Machine learning
 * project. Internal types conversion are defined by the Primitives singleton. The type
 * conversion related to each specific libraries are defined by their respective singleton
 * (i.e CommonsMath).
 *
 * @author Patrick Nicolas
 * @since 0.99.1
 * @version 0.99.2
 * @see Scala for Machine Learning - Chapter 2 Data Pipelines
 */
package object Predef {
  import Context._

  type DblPair = (Double, Double)
  type DblMatrix = Array[Array[Double]]
  type DblVec = Vector[Double]
  type DblPairVector = Vector[DblPair]
  type Features = Array[Double]
  type VSeries[T] = Vector[Array[T]]
  type DblSeries = Vector[Array[Double]]

  case class Pair(p: DblPair) {
    def +(o: Pair): Pair = Pair((p._1 + o.p._1, p._2 + o.p._2))
    def /(o: Pair): Pair = Pair((p._1 / o.p._1, p._2 / o.p._2))
  }

  final def sqr(x: Double): Double = x * x

  /**
   * Default conversion from Int to Double
   * @param n  Integer input
   * @return Double version of the integer
   */
  implicit def intToDouble(n: Int): Double = n.toDouble

  import scala.reflect.ClassTag
  implicit def t2Array[T: ClassTag](t: T): Array[T] = Array.fill(1)(t)

  implicit def arrayT2DblArray[T <: AnyVal](vt: Array[T])(implicit f: T => Double): Array[Double] =
    vt.map(_.toDouble)

  /**
   * In place division of all elements of a given row of a matrix
   * @param m Matrix of elements of type Double
   * @param row  Index of the row
   * @param z  Quotient for the division of elements of the given row
   * @throws java.lang.IllegalArgumentException If the row index is out of bounds
   */
  @throws(classOf[IllegalArgumentException])
  implicit def /(m: DblMatrix, row: Int, z: Double): Unit = {
    require(row < m.length, s"/ matrix column $row out of bounds")
    require(Math.abs(z) > 1e-32, s"/ divide column matrix by $z too small")

    m(row).indices.foreach(m(row)(_) /= z)
  }

  /**
   * Implicit conversion of a VSeries[T] to a Matrix of Double
   * @param xt  Time series of elements Array[T]
   * @tparam T  Type of elements of feature
   * @return Matrix of type Array[Array[Double]]
   */
  implicit def seriesT2Double[T: ToFeatures](xt: Vector[Array[T]]): Vector[Features] = {
    val convert: ToFeatures[T] = implicitly[ToFeatures[T]]
    xt.map(convert(_))
  }

  /**
   * Implicit conversion of a pair of pairs to a Matrix with elements of type Double
   * @param x  Pair of tuple (Double, Double)
   * @return 2x2 matrix of elements of type Double
   */
  implicit def dblPairs2DblMatrix2(x: ((Double, Double), (Double, Double))): DblMatrix =
    Array[Array[Double]](Array[Double](x._1._1, x._1._2), Array[Double](x._2._1, x._2._2))

  implicit def /(v: Features, n: Int): Try[Features] = Try(v.map(_ / n))

  /**
   * Textual representation of a vector with and without the element index
   * @param v vector to represent
   * @param index flag to display the index of the element along its value. Shown if index is
   * true, not shown otherwise
   */
  @throws(classOf[IllegalArgumentException])
  implicit def toText(v: Features, index: Boolean): String = {
    require(
      v.length > 0,
      "ScalaMl.toText Cannot create a textual representation of a undefined vector"
    )
    if (index) v.zipWithIndex.map { case (x, n) => s"$x:$n" }.mkString(", ")
    else v.mkString(", ").trim
  }

  /**
   * Textual representation of a matrix with and without the element index
   * @param m matrix to represent
   * @param index flag to display the index of the elements along their value. Shown if
   * index is true, not shown otherwise
   */
  @throws(classOf[IllegalArgumentException])
  implicit def toText(m: DblMatrix, index: Boolean): String = {
    require(
      m.length > 0,
      "ScalaMl.toText Cannot create a textual representation of a undefined vector"
    )

    if (index)
      m.zipWithIndex.map { case (v, n) => s"$n:${toText(v, index)}" }.mkString("\n")
    else
      m.map(v => s"${toText(v, index)}").mkString("\n")
  }

  /**
   * Singleton that implement context bound
   * {{{
   *   From type T to Double
   *   From Array[T] to Array[Double]
   * }}}
   */
  object Context {
    trait ToDouble[T] { def apply(t: T): Double }

    implicit val int2Double = new ToDouble[Int] {
      def apply(t: Int): Double = t.toDouble
    }

    implicit val double2Double = new ToDouble[Double] {
      def apply(t: Double): Double = t
    }

    trait ToFeatures[T] { def apply(xt: Array[T]): Features }

    final class ArrayDouble2Features extends ToFeatures[Double] {
      def apply(ar: Array[Double]): Features = ar
    }

    final class ArrayInt2Features extends ToFeatures[Int] {
      def apply(ar: Array[Int]): Features = ar.map(_.toDouble)
    }
  }
}


// ---------------------------  EOF ----------------------------------------

