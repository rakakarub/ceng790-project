package edu.metu.ceng790.project

import org.apache.spark.sql.{DataFrame, Row, SparkSession}
import DataProcess._
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.tuning.{CrossValidator, CrossValidatorModel, ParamGridBuilder}
import org.apache.spark.sql.functions.{col, udf, when}
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.regression.GBTRegressor

object ModelTrain {

  def main(args: Array[String]): Unit = {

    val spark = SparkSession.builder.appName("Ceng-790 Big Data Project")
      .config("spark.master", "local[*]")
      .config("spark.driver.memory", "4700m")
      .config("spark.executor.memory", "4700m")
      .getOrCreate()

    val sc = spark.sparkContext
    sc.setLogLevel("ERROR")

    val trainDF = spark.read.parquet(FINAL_TRAIN_DIR_FINAL)
    val testDF = spark.read.parquet(FINAL_TEST_DIR_FINAL)

    println("TRAIN COLUMN SIZE : " + trainDF.columns.size)
    println("TEST COLUMN SIZE : " + testDF.columns.size)

//    val oneHotDFTrain = convertCategoricalToOneHotEncoded(trainDF)
    val finalTrainDF = prepareForTraining(trainDF)
//      .select("SK_ID_CURR", "FEATURES", "label").cache()

//    finalTrainDF.printSchema()
//    finalTrainDF.orderBy("SK_ID_CURR").show(25)

//    val oneHotDFTest = convertCategoricalToOneHotEncoded(testDF)
    val finalTestDF = prepareForTraining(testDF)
      .select("SK_ID_CURR", "FEATURES")
      .cache()

    //Initialize model
    val gbt = new GBTRegressor()
      .setLabelCol("label")
      .setFeaturesCol("FEATURES")
      .setMaxMemoryInMB(1024)
      .setSeed(123456)
      .setMaxIter(100)

    val paramGrid = new ParamGridBuilder()
      .addGrid(gbt.maxDepth, Array(4, 5, 6, 7))
      .addGrid(gbt.maxBins, Array(16, 32, 48))
      .build()

    //Setup cross validator
    val crossValidation = new CrossValidator()
      .setEstimator(gbt)
      .setEvaluator(new RegressionEvaluator())
      .setEstimatorParamMaps(paramGrid)
      .setNumFolds(3)

    val model = crossValidation.fit(finalTrainDF)
    model.save(FINAL_DATASET_DIR + "gbtModelEngineered++")

    val modelLoaded = CrossValidatorModel.load(FINAL_DATASET_DIR + "gbtModelEngineered")

    val prediction = modelLoaded.bestModel.transform(finalTestDF)
    prediction.show(truncate = false)
    val normalized = prediction
      .withColumn("prediction", when(prediction("prediction") < 0, 0).otherwise(prediction("prediction")))

    normalized.select(
      col("SK_ID_CURR"),
      col("prediction").alias("TARGET")
    )
      .orderBy("SK_ID_CURR")
      .coalesce(1)
      .write.mode("overwrite")
      .option("header", "true")
      .csv(FINAL_DATASET_DIR + "submission++")

      }

}
