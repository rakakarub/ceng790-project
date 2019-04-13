package edu.metu.ceng790.project

import org.apache.spark.{SparkConf, SparkContext}

object test {
  def main(args: Array[String]): Unit = {

    val conf = new SparkConf()
      .setMaster("local[*]")
      .setAppName("Assignment-2_Part-1:Getting Your Own Recommendation")
      .set("spark.executor.memory", "5g")
    //Create Spark Context
    val sc = new SparkContext(conf)
    //Set log level
    sc.setLogLevel("ERROR")

    println("Hello World")
  }
}
