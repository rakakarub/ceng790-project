package edu.metu.ceng790.project

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.{OneHotEncoderEstimator, StringIndexer, VectorAssembler}
import org.apache.spark.sql.{DataFrame, SparkSession}

object DataProcess {

  val DATASET_HOME_DIR = "/cleanDataSet/home-credit-default-risk/"
  val TRAIN_DIR = DATASET_HOME_DIR + "application_train.csv"
  val TEST_DIR = DATASET_HOME_DIR + "application_test.csv"
  val BUREAU_DIR = DATASET_HOME_DIR + "bureau.csv"
  val BUREAU_BALANCE_DIR = DATASET_HOME_DIR + "bureau_balance.csv"
  val CREDIT_CARD_BALANCE_DIR = DATASET_HOME_DIR + "credit_card_balance.csv"
  val INSTALLMENTS_PAYMENTS_DIR = DATASET_HOME_DIR + "installments_payments.csv"
  val POS_CASH_BALANCE_DIR = DATASET_HOME_DIR + "POS_CASH_balance.csv"
  val PREVIOUS_APPLICATION_DIR = DATASET_HOME_DIR + "previous_application.csv"


  def main(args: Array[String]): Unit = {

    val spark = SparkSession.builder.appName("Ceng-790 Big Data Project").config("spark.master", "local[*]").getOrCreate()
    val sc = spark.sparkContext
    sc.setLogLevel("ERROR")


  }
  // Get only the features of the dataSet in to one column, to be able to train a model
  def prepareForTraining(dataFrame : DataFrame): Unit = {
    val newDataFrame = new VectorAssembler()
      .setInputCols(dataFrame.columns.filterNot(e => e == "label").filterNot(f => f == "SK_ID_CURR"))
      .setOutputCol("FEATURES")
      .transform(dataFrame)

    newDataFrame
  }

  def convertCategoricalToOneHotEncoded(dataFrame: DataFrame) : DataFrame = {

    //Find Categorical Columns
    val categoricalColumns = dataFrame
      .dtypes
      .filter(e => e._2.startsWith("String"))
      .map(f => f._1)

    //Create StringIndexer
    val columnStringIndexer = categoricalColumns.map({
      colunmName =>
        new StringIndexer()
          .setInputCol(colunmName)
          .setOutputCol(colunmName + "_STRING_INDEXED")
          .setHandleInvalid("keep")
    })

    //Convert string idexed to one-hot
    val oneHotColumns = new OneHotEncoderEstimator()
      .setInputCols(categoricalColumns.map(_ + "_STRING_INDEXED"))
      .setOutputCols(categoricalColumns.map(_ + "_ONEHOT_ENCODED"))

    //Create pipeline
    val pipeline = new Pipeline()
      .setStages(columnStringIndexer ++ Array(oneHotColumns))

    //Apply pipeline
    val encodedDataFrame = pipeline
      .fit(dataFrame)
      .transform(dataFrame)

    //Remove string indexed columns
    val selectedColumns = encodedDataFrame
      .columns
      .filterNot(categoricalColumns contains _ )
      .filterNot(_ endsWith "_STRING_INDEXED")

    val result = encodedDataFrame.select(selectedColumns.head, selectedColumns.tail : _*)

    result
  }
}
