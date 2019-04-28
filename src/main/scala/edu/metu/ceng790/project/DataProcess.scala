package edu.metu.ceng790.project

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.{OneHotEncoderEstimator, StringIndexer, VectorAssembler}
import org.apache.spark.sql.{Column, DataFrame, SparkSession}
import org.apache.spark.sql.functions._

object DataProcess {

  val FINAL_DATASET_DIR = "finalDataSets/"
  val DATASET_HOME_DIR = "cleanDataSet/home-credit-default-risk/"
  val TRAIN_DIR = DATASET_HOME_DIR + "application_train.csv"
  val TEST_DIR = DATASET_HOME_DIR + "application_test.csv"
  val BUREAU_DIR = DATASET_HOME_DIR + "bureau.csv"
  val BUREAU_BALANCE_DIR = DATASET_HOME_DIR + "bureau_balance.csv"
  val CREDIT_CARD_BALANCE_DIR = DATASET_HOME_DIR + "credit_card_balance.csv"
  val INSTALLMENTS_PAYMENTS_DIR = DATASET_HOME_DIR + "installments_payments.csv"
  val POS_CASH_BALANCE_DIR = DATASET_HOME_DIR + "POS_CASH_balance.csv"
  val PREVIOUS_APPLICATION_DIR = DATASET_HOME_DIR + "previous_application.csv"
  val COMBINED_DATASET_DIR = FINAL_DATASET_DIR + "combinedDataSet.csv"
  val FINAL_TRAIN_DIR = FINAL_DATASET_DIR + "finalTrain.csv"
  val FINAL_TEST_DIR = FINAL_DATASET_DIR + "finalTest.csv"

  val AGGREGATE_MAPPING: Map[String, Column => Column] = Map(
    "MIN" -> min,
    "MAX" -> max,
    "MEAN" -> mean,
    "VAR" -> var_samp,
    "SUM" -> sum)

  val AGGREGATE_STRING: Map[String, Column => Column] = Map(
    "FIRST" -> first
  )

  def main(args: Array[String]): Unit = {

    val spark = SparkSession.builder.appName("Ceng-790 Big Data Project")
      .config("spark.master", "local[*]")
      .config("spark.driver.memory", "4g")
      .config("spark.executor.memory", "4g")
      .getOrCreate()

    val sc = spark.sparkContext
    sc.setLogLevel("ERROR")

    //combineDataSets
//  comineDataSet(spark)
//

    val combinedDataSet = loadFile(COMBINED_DATASET_DIR, spark)

    //Extract train data
    val trainDF : DataFrame = combinedDataSet.where(col("label").=!=(-23))

    //Extract test data
    val testDF : DataFrame = combinedDataSet.where(col("label").===(-23)).drop("label")

    println("Total rows : " + combinedDataSet.count())
    println("Total columns : " + combinedDataSet.columns.size)

    println("Train rows : " + trainDF.count())
    println("Train columns : " + trainDF.columns.size)

    println("Test rows : " + testDF.count())
    println("Test columns : " + testDF.columns.size)

    trainDF.coalesce(1).write
      .format("csv")
      .option("header", "true")
      .save(FINAL_DATASET_DIR + "finalTrain.csv")

    testDF.coalesce(1).write
      .format("csv")
      .option("header", "true")
      .save(FINAL_DATASET_DIR + "finalTest.csv")
//    val oneHotDFTrain = convertCategoricalToOneHotEncoded(trainDF)
//    val finalTrainDF = prepareForTraining(oneHotDFTrain)
//
//    val oneHotDFTest = convertCategoricalToOneHotEncoded(testDF)
//    val finalTestDF = prepareForTraining(oneHotDFTest)


  }

  def comineDataSet(spark: SparkSession) : Unit = {
    //Load test/train data
    val trainDF = loadFile(TRAIN_DIR, spark)
    val testDF = loadFile(TEST_DIR, spark)

    //Union train and test data
    val unionDF = trainDF.union(testDF)

    //prepare/normalize the tables
    val prevApplicationDF = prevAppNormalize(PREVIOUS_APPLICATION_DIR, spark)
    val posCashDF = posCashNormalize(POS_CASH_BALANCE_DIR, spark)
    val installmentsDF = installmentPaymentsNormalize(INSTALLMENTS_PAYMENTS_DIR, spark)
    val creditCardDF = creditCardDataNormalize(CREDIT_CARD_BALANCE_DIR, spark)
    val bureauDF = bureauBalanceNormalize(spark)


    //Combine all tables into one
    val combinedDataSet = List(
      unionDF,
      prevApplicationDF,
      posCashDF,
      installmentsDF,
      creditCardDF,
      bureauDF
    )
      .reduce(
        (a : DataFrame, b : DataFrame) =>
          a.join(b, Seq("SK_ID_CURR"), "left")
      )

    combinedDataSet.coalesce(1).write
      .format("csv")
      .option("header", "true")
      .save(FINAL_DATASET_DIR + "combinedDataSet.csv")
  }
  // Get only the features of the dataSet in to one column, to be able to train a model
  def prepareForTraining(dataFrame : DataFrame): DataFrame = {
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

    println("String Indexer Size : " + categoricalColumns.size)

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

  def creditCardDataNormalize(dataDirectory: String, sparkSession: SparkSession) : DataFrame = {
    var creditCardDataSet = loadFile(CREDIT_CARD_BALANCE_DIR, sparkSession)
      .drop("SK_ID_PREV")

    val stringColumns = findNumericalAndCategoricalFeatures(creditCardDataSet)._1
    val stringColumnsAndModes = findMostOccuredValueInAColumn(creditCardDataSet, stringColumns)

   val creaditCardAggregates = getCreaditCardAggregates(creditCardDataSet)

    creditCardDataSet = myAggragate(creditCardDataSet, Seq("SK_ID_CURR"), creaditCardAggregates ++ getStringAggregates(creditCardDataSet))



    creditCardDataSet
  }

  def bureauBalanceNormalize(sparkSession: SparkSession) : DataFrame = {
    val dataSetBalance = loadFile(BUREAU_BALANCE_DIR, sparkSession)
    val dataSetBureau = loadFile(BUREAU_DIR, sparkSession)
      .withColumnRenamed("AMT_ANNUITY", "AMT_ANNUITY_BUREAU")

    val expressions = Seq("MONTHS_BALANCE")
      .flatMap(
        currentColumn => AGGREGATE_MAPPING.map(
          aggOperation => AGGREGATE_MAPPING(aggOperation._1)(col(currentColumn)).alias(currentColumn + "_" + aggOperation._1)
        )
      )

    var finalBalance = dataSetBalance.groupBy("SK_ID_BUREAU")
      .agg(expressions.head, expressions.tail : _*)

    finalBalance = finalBalance
      .join(dataSetBureau, Seq("SK_ID_BUREAU")).drop("SK_ID_BUREAU")

    finalBalance = myAggragate(finalBalance, Seq("SK_ID_CURR"), getBureauAggregates())
    finalBalance
  }

  def installmentPaymentsNormalize(dataDirectory: String, sparkSession: SparkSession): DataFrame = {
    val installmentDataSet = loadFile(INSTALLMENTS_PAYMENTS_DIR, sparkSession)
      .drop("SK_ID_PREV")

    val aggregations = getInstallmentAggregates()
    installmentDataSet.printSchema()

    val newDataSet = myAggragate(installmentDataSet, Seq("SK_ID_CURR"), aggregations)

    newDataSet
  }

  def posCashNormalize(dataDirectory: String, sparkSession: SparkSession): DataFrame = {
    val posCashDataSet = loadFile(POS_CASH_BALANCE_DIR, sparkSession)
      .withColumnRenamed("MONTHS_BALANCE", "MONTHS_BALANCE_POS_CASH")
      .withColumnRenamed("SK_DPD", "SK_DPD_POS_CASH")
      .withColumnRenamed("SK_DPD_DEF", "SK_DPD_DEF_POS_CASH")

    val newDataSet = myAggragate(posCashDataSet, Seq("SK_ID_CURR"), getPosCashAggregates())
    newDataSet
  }

  def prevAppNormalize(dataDirectory: String, sparkSession: SparkSession): DataFrame = {
    val prevAppDataSet = loadFile(PREVIOUS_APPLICATION_DIR, sparkSession)
      .withColumnRenamed("AMT_ANNUITY", "AMT_ANNUITY_PREV_APP")
      .withColumnRenamed("NAME_CONTRACT_STATUS", "NAME_CONTRACT_STATUS_PREV_APP")

    val newDataSet = myAggragate(prevAppDataSet, Seq("SK_ID_CURR"), getPreviousApplicationAggregates() ++ getStringAggregates(prevAppDataSet))
    newDataSet

  }

  def loadFile(fileName : String, sparkSession: SparkSession) : DataFrame = {
    val dataSet = sparkSession.sqlContext.read
      .format("csv")
      .option("header", true)
      .option("inferSchema", true)
      .load(fileName)

    dataSet
  }

  def myAggragate(dataSet: DataFrame, groupByKey: Seq[String], columns: Map[String, Seq[Column => Column]]): DataFrame = {
    val expressions = columns.flatMap(
      c => c._2
        .zipWithIndex
        .map {
          case (method, index) =>
            method(col(c._1)).alias(c._1 + "_MTD" + (index + 1).toString)
        }).toList

      dataSet.groupBy(groupByKey.map(col): _*).agg(expressions.head, expressions.tail: _*)
  }

  def getBureauAggregates(): Map[String, Seq[Column => Column]] ={
    val aggregateMap : Map[String, Seq[Column => Column]] = Map(
      "DAYS_CREDIT" -> Seq(mean, var_samp),
      "DAYS_CREDIT_ENDDATE" -> Seq(mean),
      "DAYS_CREDIT_UPDATE" -> Seq(mean),
      "CREDIT_DAY_OVERDUE" -> Seq(mean),
      "AMT_CREDIT_MAX_OVERDUE" -> Seq(mean),
      "AMT_CREDIT_SUM" -> Seq(mean, sum),
      "AMT_CREDIT_SUM_DEBT" -> Seq(mean, sum),
      "AMT_CREDIT_SUM_OVERDUE" -> Seq(mean),
      "AMT_CREDIT_SUM_LIMIT" -> Seq(mean, sum),
      "AMT_ANNUITY_BUREAU" -> Seq(max, mean),
      "CNT_CREDIT_PROLONG" -> Seq(sum),
      "MONTHS_BALANCE_MIN" -> Seq(min),
      "MONTHS_BALANCE_MAX" -> Seq(max),
      "MONTHS_BALANCE_MEAN" -> Seq(mean),
      "MONTHS_BALANCE_SUM" -> Seq(sum),
      "MONTHS_BALANCE_VAR" -> Seq(mean))
    aggregateMap
  }

  def getCreaditCardAggregates(dataFrame: DataFrame): Map[String, Seq[Column => Column]] = {
    val aggregationMap = dataFrame.columns.filterNot(_ startsWith "SK_ID_").filterNot(_ endsWith "_STRING_INDEXED").map(
      c => c -> AGGREGATE_MAPPING.values.toSeq
    ).toMap

    aggregationMap
  }

  def getInstallmentAggregates(): Map[String, Seq[Column => Column]] = {
    val aggregates: Map[String, Seq[Column => Column]] = Map(
      "NUM_INSTALMENT_VERSION" -> Seq(max, mean, sum, min, stddev_samp),
      "NUM_INSTALMENT_NUMBER" -> Seq(max, mean, variance, min, stddev_samp),
      "DAYS_INSTALMENT" -> Seq(max, mean, variance, min, stddev_samp),
      "AMT_INSTALMENT" -> Seq(max, mean, sum, min, stddev_samp),
      "AMT_PAYMENT" -> Seq(min, max, mean, sum, stddev_samp),
      "DAYS_ENTRY_PAYMENT" -> Seq(max, mean, sum, stddev_samp)
    )

    aggregates
  }

  def getPosCashAggregates(): Map[String, Seq[Column => Column]] = {
    val aggregates : Map[String, Seq[Column => Column]] = Map(
      "MONTHS_BALANCE_POS_CASH" -> Seq(max, mean),
      "SK_DPD_POS_CASH" -> Seq(max, mean),
      "SK_DPD_DEF_POS_CASH" -> Seq(max, mean)
    )
    aggregates
  }

  def getPreviousApplicationAggregates(): Map[String, Seq[Column => Column]] = {
    val aggregates : Map[String, Seq[Column => Column]] = Map(
      "AMT_ANNUITY_PREV_APP" -> Seq(max, mean),
      "AMT_APPLICATION" -> Seq(max, mean),
      "AMT_CREDIT" -> Seq(max, mean),
      "AMT_DOWN_PAYMENT" -> Seq(max, mean),
      "AMT_GOODS_PRICE" -> Seq(max, mean),
      "HOUR_APPR_PROCESS_START" -> Seq(max, mean),
      "RATE_DOWN_PAYMENT" -> Seq(max, mean),
      "DAYS_DECISION" -> Seq(max, mean),
      "CNT_PAYMENT" -> Seq(mean, sum)
    )

    aggregates
  }

  def findMostOccuredValueInAColumn(dataFrame: DataFrame, categoricalFeatures : Seq[String]): Map[String, String] = {
    val categoricalColumnsAndMosts : Map[String, String] = categoricalFeatures.map(f => {
      val mode : String = dataFrame.groupBy(f).count().sort(col("count").desc).collect()(0)(0).toString
      (f, mode)
    }).toMap
    categoricalColumnsAndMosts
  }

  def findNumericalAndCategoricalFeatures(dataSet : DataFrame): (Array[String], Array[String]) = {
    val categoricalColumns = dataSet.dtypes.filter(e => e._2.startsWith("String") && !e._1.startsWith("label") && !e._1.startsWith("SK_ID_CURR")).map(a => a._1)
    val numericalColumns = dataSet.dtypes.filter(e => (e._2.startsWith("Integer") || e._2.startsWith("Double")) && !e._1.startsWith("label") && !e._1.startsWith("SK_ID_CURR")).map(a => a._1)

    println("Categorical Column Size : " + categoricalColumns.size)
    println("Numerical Column Size : " + numericalColumns.size)

    (categoricalColumns, numericalColumns)
  }

  def getStringAggregates(dataFrame: DataFrame): Map[String, Seq[Column => Column]] = {
    val stringColumns = dataFrame.dtypes.filter(f => f._2 == "StringType").map(f => f._1)
    val aggs: Map[String, Seq[Column => Column]] =

    stringColumns
      .map(c => c -> Seq(AGGREGATE_STRING("FIRST"))) // For some reason it doesn't work to put just `mean` here.
      .toMap

    aggs
  }

}
