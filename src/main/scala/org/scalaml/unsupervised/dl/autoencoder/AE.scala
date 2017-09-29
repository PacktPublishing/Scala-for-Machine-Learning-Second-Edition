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

import org.apache.log4j.Logger
import org.scalaml.Predef.Context.ToDouble
import org.scalaml.core.ITransform
import org.scalaml.stats.TSeries.dimension
import org.scalaml.supervised.mlp.MLP.MLPMode
import org.scalaml.supervised.mlp.MLPModel
import org.scalaml.supervised.mlp.MLPModel
import org.scalaml.util.LoggingUtils.Monitor
import org.scalaml.util.MathUtils._

import scala.annotation.implicitNotFound
import scala.util.Try


/**
  * Implementation of Auto-encoder. The configuration of the auto-encoder is defined as the input and output layer
  * (input and output layers having the same number of neurons), the representation (compression) layer and the
  * intermediate hidden layer prior and after the representation layer.
  * The representation layer is not considered as part of the hidden layer. For instance, a 5-layer auto-encoder will
  * be defined as
  *  - input layer (x neurons)
  *  - hidden layer (y neurons)
  *  - representation layer (z neurons)
  *  - hidden layer (y neurons)
  *  - output layer (x neurons)
  * @author Patrick Nicolas
  * @version 0.99.2
  *
  * @param config Configuration parameters used by the auto-encoder
  * @param hidden Configuration of intermediate hidden layers beside the representation layer
  * @param representation Number of neurons in the representation layer
  * @param xt Input data or observations
  * @tparam T  Type of input to the data transformation
  * @see Scala for Machine Learning Chapter 11 Deep learning / Sparse Auto-encoder
  */
@throws(classOf[IllegalArgumentException])
@implicitNotFound(msg = "AE implicit conversion to Double undefined")
final private[scalaml] class AE[@specialized(Double) T: ToDouble] protected (
    config: AEConfig,
    hidden: Array[Int],
    representation: Int,
    xt: Vector[Array[T]]) extends ITransform[Array[T], Array[Double]] with Monitor[Double] {
  import AE._

  require(xt.size > 0, "Autoencoder cannot process undefined data set")
  protected val logger = Logger.getLogger("AE")
  /**
    * Initializes the topology of this multi-layer perceptron starting
    * from the input data, the output and the symmetric hidden layers.
    */
  lazy val topology =
    if (hidden.length == 0)
      Array[Int](xt.head.length, representation, xt.head.length)
    else
      Array[Int](xt.head.length) ++ generateHiddenLayer(hidden, representation) ++ Array[Int](xt.head.length)

  /**
    * Model for the Multi-layer Perceptron of type MLPNetwork. This implementation
    * allows the model to be created even in the training does not converged towards
    * a stable network of synapse weights. The client code is responsible for
    * evaluating the value of the state variable converge and perform a validation run
    */
  val model: Option[MLPModel] =  {
    import Shuffle._, Math._

    val network = AENetwork(config, topology)
    var prevErr = Double.MaxValue

    (0 until config.numEpochs).find(n => {
      val err = fisherYates(xt.size)
          .map(n => {
            val z: Array[Double] = xt(n).map(implicitly[ToDouble[T]].apply(_))
            network.trainEpoch(z, z)
          }).sum / xt.size

      val diffErr = err - prevErr
      prevErr = err
      abs(diffErr) < config.eps
    }).map(_ => network.getModel)
  }

  /**
    * Define the predictive function of the classifier or regression as a data
    * transformation by overriding the pipe operator |>.
    * @throws MatchError if the model is undefined or the input string has an incorrect size
    * @return PartialFunction of features vector of type Array[T] as input and
    * the predicted vector values as output
    */
  override def |> : PartialFunction[Array[T], Try[Array[Double]]] = {
    case x: Array[T] if  model.isDefined && x.length == dimension(xt) =>
      Try(AENetwork(config, topology, model).predict(x.map(implicitly[ToDouble[T]].apply(_))))
  }
}


/**
  * Companion object used for the generation of the layout of the network (with hidden and representation layer)
  * and the Sparse MSE mode for compatibility with the multi-layer perceptron
  * @author Patrick Nicolas
  * @version 0.99.2
  */
private[scalaml] object AE {
  import org.scalaml.stats.Loss._

  implicit val sparseMSE: SparseMSE = new SparseMSE

  /**
    * Perform configuration of the hidden layer using a sparse representation.
    * @param hidden Configuration of the hidden layers beside the representation layer
    * @param representation Number of node in the representation layer
    * @return Configuration of all the hidden layer including the representation layer
    */
  protected def generateHiddenLayer(hidden: Array[Int], representation: Int): Array[Int] =
    if(hidden.nonEmpty ||
      ((1 until hidden.size).forall(n => hidden(n) >= hidden(n+1)) && representation < hidden.last))
      hidden ++ Array[Int](representation) ++ hidden.reverse
    else
      Array.empty[Int]

  final class SparseMSE extends MLPMode {
    /**
      * Normalize the output vector to match the objective of the autoencoder.
      */
    override def apply(output: Array[Double]): Array[Double] = output.map(sigmoid(_))

    override def error(expected: Array[Double], output: Array[Double]): Double = mse(expected, output)
  }

  def apply[T: ToDouble](
    config: AEConfig,
    hidden: Array[Int],
    representation: Int,
    xt: Vector[Array[T]]) = new AE(config, hidden, representation, xt)

  def apply[T: ToDouble](
    config: AEConfig,
    representation: Int,
    xt: Vector[Array[T]]) = new AE(config, Array.empty[Int], representation, xt)
}


// -----------------------------------------  EOF --------------------------------------------------------------------


