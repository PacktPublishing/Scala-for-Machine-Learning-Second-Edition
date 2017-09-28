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
package org.scalaml.supervised.bayes.text

import scala.collection._
import scala.util.Try
import org.scalaml.core.ETransform
import org.scalaml.workflow.data.DocumentsSource
import org.scalaml.util.MapUtils.Counter
import DocumentsSource._
import TextAnalyzer._
import org.scalaml.core.Design.Config

/**
 * Simple text analyzer that extract the relative frequencies of keywords defined in a
 * lexicon. The text analyzer is implemented as a data transformation using an explicit
 * configuration (Lexicon) '''ETransform'''
 * {{{
 *   1. Parse content of each document
 *   2. Counts the number of occurrences for a select set of keywords from each document
 *   3. Aggregate the count of occurrences of keywords
 *   4. Compute the relative frequency for the set of keywords
 * }}}
 * @tparam T type of date or time stamp of documents
 * @constructor Create a text analyzer with a given parser and lexicon
 * @param parser Basic document/text parser
 * @param lexicon semantic or synonyms map
 * @author Patrick Nicolas
 * @since 0.99  June 17, 2015
 * @version 0.99.2
 * @see Scala for Machine Learning Chapter 5 "Naive Bayes Models" / Naive Bayes and text mining
 * @see org.scalaml.core.ETransform
 */
class TextAnalyzer[T <: AnyVal](
    parser: TextParser,
    lexicon: Lexicon
)(implicit ordering: Ordering[T], f: T => Long) extends ETransform[Corpus[T], Seq[TermsRF]](lexicon) {

  override def |> : PartialFunction[Corpus[T], Try[Seq[TermsRF]]] = {
    case docs: Corpus[T] => Try(score(docs))
  }

  private def score(corpus: Corpus[T]): Seq[TermsRF] = {

    // Count the occurrences of news article for each specific date
    val termsCount: Seq[(T, Counter[String])] = corpus.map(doc => (doc.date, count(doc.content)))

    // Aggregate the term count for all documents related to each date
    val termsCountMap: Map[T, Counter[String]] =
      termsCount.groupBy(_._1).map {
        case (t, seq) =>
          (t, seq.aggregate(new Counter[String])((s, cnt) => s ++ cnt._2, _ ++ _))
      }

    // Sort and reformat the term counts per date
    val termsCountPerDate = termsCountMap.toSeq.sortBy(_._1).unzip._2

    // Compute the terms count for the entire corpus
    val allTermsCount = termsCountPerDate.aggregate(new Counter[String])((s, cnt) => s ++ cnt, _ ++ _)
    // Computes the relative (normalized) frequencies of each terms.
    termsCountPerDate.map(_ / allTermsCount).map(_.toMap)
  }

  def quantize(termsRFSeq: Seq[TermsRF]): Try[(Array[String], Vector[Array[Double]])] = Try {
    val keywords: Array[String] = lexicon.values.distinct

    val features: Seq[Array[Double]] = termsRFSeq.map(tf =>
      keywords.map(key => tf.getOrElse(key, 0.0)))
    (keywords, features.toVector)
  }

  /**
   * Count the number of occurrences of a term. The function toWords
   * extract the keywords matching the lexicon from a string or file entry.
   */
  private def count(term: String): Counter[String] = {
    require(
      term.length > 0,
      "TermsScore.count: Cannot count the number of words in undefined text"
    )
    parser(term)./:(new Counter[String])((cnt, w) =>
      if (lexicon.contains(w)) cnt + w else cnt)
  }
}

case class Lexicon(dictionary: immutable.Map[String, String]) extends Config {
  final def values: Array[String] = dictionary.values.toArray
  final def contains(key: String): Boolean = dictionary.contains(key)
  def apply(key: String): Int = if (dictionary.contains(key)) 1 else 0
}

object TextAnalyzer {
  type TermsRF = Map[String, Double]
  type TextParser = String => Array[String]

  def apply[T <: AnyVal](
    parser: TextParser,
    lexicon: Lexicon
  )(implicit f: T => Long, ordering: Ordering[T]): TextAnalyzer[T] = new TextAnalyzer[T](parser, lexicon)

}

// ----------------------------  EOF ------------------------------------------