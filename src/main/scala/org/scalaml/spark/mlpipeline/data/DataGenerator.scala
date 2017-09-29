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
package org.scalaml.spark.mlpipeline.data

/**
  * Data generator command line application
  * @author Patrick Nicolas
  * @version 0.99.2
  */
private[spark] object DataGenerator extends App {
  val entries = Seq[Array[String]](
    Array[String]("07/10/2014", "07/09/2014", "11/24/2014", "10/18/2014", "09/11/2014", "08/03/2014", "09/28/2014", "10/18/2014", "06/06/2014", "07/05/2014", "09/02/2014", "07/09/2014", "11/07/2014", "07/10/2014", "10/25/2014", "09/10/2014", "10/01/2014", "06/07/2014", "11/20/2014"),
    Array[String]("309ae914", "28bb005c", "300ab097", "207ee09a", "22000bc6", "2981aa92", "26301ab4", "240abc81", "2101bef2", "2305bce3", "300ec90b", "23c9a89d", "2820acd8", "270c0eb1", "29fe9013", "28bb005c"),
    Array[String]("28", "15", "18", "22", "24", "26", "13", "09", "27", "12", "25", "33", "30", "11", "14", "12", "07", "03", "04", "01", "14", "15", "22", "23"),
    Array[String]("aa5", "56b", "90d", "a08", "c07", "819", "bc0", "89f", "1ac", "227", "bc7", "cc2", "d95", "101", "da6", "74f")
  )

  def nextEntry(n: Int): String = entries(n)(scala.util.Random.nextInt(entries(n).length))
  def nextLabel(index: Int): String = if (index % 3 == 0) "1.0" else "0.0"
  def nextEntries: String = {
    s"${nextEntry(0)},${nextEntry(1)},${nextEntry(2)},${nextEntry(3)},${nextLabel(scala.util.Random.nextInt(2))}"
  }

  var printWriter: java.io.PrintWriter = _
  try {
    printWriter = new java.io.PrintWriter("chap17_streaming_input2.csv")
    printWriter.println("date,asset,region,agent,label")
    (0 until 100000).foreach(_ => printWriter.println(nextEntries))
    printWriter.close
  } finally {
    try {
      if (printWriter != null)
        printWriter.close
    } catch {
      case e: java.io.IOException => println("error")
    }
  }
}
