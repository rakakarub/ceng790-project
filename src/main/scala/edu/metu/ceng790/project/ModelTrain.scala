package edu.metu.ceng790.project

import org.apache.spark.sql.SparkSession
import DataProcess._
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.regression.GBTRegressor
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}

object ModelTrain {

  def main(args: Array[String]): Unit = {

    val spark = SparkSession.builder.appName("Ceng-790 Big Data Project")
      .config("spark.master", "local[*]")
      .config("spark.driver.memory", "4g")
      .config("spark.executor.memory", "4g")
      .getOrCreate()

    val sc = spark.sparkContext
    sc.setLogLevel("ERROR")

    val trainDF = loadFile(FINAL_TRAIN_DIR, spark).na.fill(0).repartition(100)
    val testDF = loadFile(FINAL_TEST_DIR, spark).na.fill(0).repartition(100)

    val oneHotDFTrain = convertCategoricalToOneHotEncoded(trainDF)
    val finalTrainDF = prepareForTraining(oneHotDFTrain).cache()

    val oneHotDFTest = convertCategoricalToOneHotEncoded(testDF)
    val finalTestDF = prepareForTraining(oneHotDFTest).cache()

    //Initialize model
    val gbt = new GBTRegressor()
      .setLabelCol("label")
      .setFeaturesCol("FEATURES")
      .setMaxIter(100)

    val paramGrid = new ParamGridBuilder()
      .build()

    //Setup cross validator
    val crossValidation = new CrossValidator()
      .setEstimator(gbt)
      .setEvaluator(new RegressionEvaluator())
      .setEstimatorParamMaps(paramGrid)
      .setNumFolds(3)

    val model = crossValidation.fit(finalTrainDF)
    model.save(FINAL_DATASET_DIR + "gbtModel")

    val prediction = model.bestModel.transform(finalTestDF)
    prediction.printSchema()


  }

}
