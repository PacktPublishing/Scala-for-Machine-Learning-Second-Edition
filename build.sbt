organization := "Patrick Nicolas"

name := "Scala for Machine Learning - 2nd Edition"

version := "0.99.2"

scalaVersion := "2.11.8"

libraryDependencies ++= Seq(
  "org.apache.commons" % "commons-math3" % "3.6",
  "org.jfree" % "jfreechart" % "1.0.17",
  "com.typesafe.akka" %% "akka-actor" % "2.3.8",
  "org.apache.spark" %% "spark-core" % "2.1.0",
  "org.apache.spark" %% "spark-mllib" % "2.1.0",
  "org.apache.spark" %% "spark-streaming" % "2.1.0",
  "org.scalatest" %% "scalatest" % "2.2.6"
)

// Resolver for Apache Spark framework
resolvers += "Akka Repository" at "http://repo.akka.io/releases/"

// Options for the Scala compiler should be customize
scalacOptions ++= Seq("-unchecked", "-optimize", "-language:postfixOps")

// Specifies the content for the root package used in Scaladoc
scalacOptions in (Compile, doc) ++= Seq("-doc-root-content", baseDirectory.value + "/root-doc.txt")

excludeFilter in doc in Compile := "*.java" || "*.jar"

// Set the path to the unmanaged, 3rd party libraries
unmanagedClasspath in Compile ++= Seq(
  file("lib/CRF-1.1.jar"),
  file("lib/Trove-3.0.2.jar"),
  file("lib/colt.jar"),
  file("lib/libsvm_sml-3.18.jar")
)

// Options for Scala test with timing and short stack trace
testOptions in Test += Tests.Argument("-oDS")

// Maximum number of errors during build
maxErrors := 30

