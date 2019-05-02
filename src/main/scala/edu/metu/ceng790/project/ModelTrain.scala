package edu.metu.ceng790.project

import org.apache.spark.sql.SparkSession
import DataProcess._
import org.apache.spark.ml.tuning.{CrossValidator, CrossValidatorModel, ParamGridBuilder}

object ModelTrain {

  def main(args: Array[String]): Unit = {

    val spark = SparkSession.builder.appName("Ceng-790 Big Data Project")
      .config("spark.master", "local[*]")
      .config("spark.driver.memory", "4g")
      .config("spark.executor.memory", "4g")
      .getOrCreate()

    val sc = spark.sparkContext
    sc.setLogLevel("ERROR")

//    val trainDF = loadFile(FINAL_TRAIN_DIR, spark).na.fill(0).repartition(100)
    val testDF = loadFile(FINAL_TEST_DIR, spark).na.fill(0)
//
//    val oneHotDFTrain = convertCategoricalToOneHotEncoded(trainDF)
//    val finalTrainDF = prepareForTraining(oneHotDFTrain).cache()
//
    val oneHotDFTest = convertCategoricalToOneHotEncoded(testDF)
    val finalTestDF = prepareForTraining(oneHotDFTest).cache()
//
//    //Initialize model
//    val gbt = new GBTRegressor()
//      .setLabelCol("label")
//      .setFeaturesCol("FEATURES")
//      .setMaxIter(100)
//
//    val paramGrid = new ParamGridBuilder()
//      .build()
//
//    //Setup cross validator
//    val crossValidation = new CrossValidator()
//      .setEstimator(gbt)
//      .setEvaluator(new RegressionEvaluator())
//      .setEstimatorParamMaps(paramGrid)
//      .setNumFolds(3)
//
//    val model = crossValidation.fit(finalTrainDF)
//    model.save(FINAL_DATASET_DIR + "gbtModel2")

    val model = CrossValidatorModel.load(FINAL_DATASET_DIR + "gbtModel2")

    val prediction = model.bestModel.transform(finalTestDF)
    prediction.printSchema()

    import org.apache.spark.sql.functions.{udf, col}
    import org.apache.spark.ml.linalg.Vector
    val second = udf((v : Vector) => v.toArray(1))

//    val result = prediction
//      .select(col("SK_ID_CURR"),
//        second(col("prediction")).alias("TARGET"))
//      .orderBy("SK_ID_CURR")
//        .coalesce(1)
//        .write
//        .format("csv")
//        .option("header", "true")
//        .save(FINAL_DATASET_DIR + "results.csv")

    val columnNames = Seq("SK_ID_CURR", "prediction")
    val result = prediction.select("SK_ID_CURR", "prediction").toDF().cache()
    println("Size : " + result.count())

    val orderedDF = result.orderBy(col("SK_ID_CURR").asc).cache()
    println("Sizee: " + orderedDF.count())
//      .orderBy("SK_ID_CURR")
//      .show(5)
//      .coalesce(1)
//      .write
//      .format("csv")
//      .option("header", "true")
//      .save(FINAL_DATASET_DIR + "results.csv")



  }

}
