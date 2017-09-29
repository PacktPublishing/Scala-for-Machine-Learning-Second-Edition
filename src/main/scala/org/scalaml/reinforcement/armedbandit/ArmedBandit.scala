package org.scalaml.reinforcement.armedbandit

import org.scalaml.core.ITransform

import scala.util.Try


/**
  * Generic Armed bandit interface
  * @author Patrick Nicolas
  * @version 0.99.2
  * @tparam U Type of the arm
  */
private[scalaml] trait ArmedBandit[U <: Arm] extends ITransform[U, Double]{
  protected[this] val arms: List[U] = List.empty[U]
  protected[this] var cumulRegret: Double = _

  def select: U

  def apply(successArm: U): Unit

  /**
    *
    * @return A partial function that implement the conversion of data element T => Try[A]
    */
  override def |> : PartialFunction[U, Try[Double]] = {
    case successArm: U @unchecked if(successArm.successes + successArm.failures > 1) =>
      val playedArm = select
      this(successArm)
      cumulRegret += playedArm.mean - successArm.mean
      Try(cumulRegret)
  }
}


// ------------------------------  EOF --------------------------------------------------
