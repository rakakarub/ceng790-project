package edu.metu.ceng790.project

import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.{sql}
import org.apache.spark.sql.functions._

object dataCleaner {

  val DATASET_HOME_DIR = "home-credit-default-risk/"
  val TRAIN_DIR = DATASET_HOME_DIR + "application_train.csv"
  val TEST_DIR = DATASET_HOME_DIR + "application_test.csv"
  val BUREAU_DIR = DATASET_HOME_DIR + "bureau.csv"
  val BUREAU_BALANCE_DIR = DATASET_HOME_DIR + "bureau_balance.csv"
  val CREDIT_CARD_BALANCE_DIR = DATASET_HOME_DIR + "credit_card_balance.csv"
  val INSTALLMENTS_PAYMENTS_DIR = DATASET_HOME_DIR + "installments_payments.csv"
  val POS_CASH_BALANCE_DIR = DATASET_HOME_DIR + "POS_CASH_balance.csv"
  val PREVIOUS_APPLICATION_DIR = DATASET_HOME_DIR + "previous_application.csv"

  val TRAIN_DATA = 1;
  val TEST_DATA = 2;
  val SUPPLEMENTARY_DATA = 3;

  def main(args: Array[String]): Unit = {

    val spark = SparkSession.builder.appName("Ceng-790 Big Data Project").config("spark.master", "local[*]").getOrCreate()
    val sc = spark.sparkContext
    sc.setLogLevel("ERROR")

    cleanDataSet(TRAIN_DIR, spark, TRAIN_DATA)
    cleanDataSet(TEST_DIR, spark, TEST_DATA)
    cleanDataSet(BUREAU_DIR, spark, SUPPLEMENTARY_DATA)
    cleanDataSet(BUREAU_BALANCE_DIR, spark, SUPPLEMENTARY_DATA)
    cleanDataSet(CREDIT_CARD_BALANCE_DIR, spark, SUPPLEMENTARY_DATA)
    cleanDataSet(INSTALLMENTS_PAYMENTS_DIR, spark, SUPPLEMENTARY_DATA)
    cleanDataSet(POS_CASH_BALANCE_DIR, spark, SUPPLEMENTARY_DATA)
    cleanDataSet(PREVIOUS_APPLICATION_DIR, spark, SUPPLEMENTARY_DATA)
  }

  def cleanDataSet(dataSetDirectory : String, sparkSession: SparkSession, dataType : Int): Unit = {

    var ds = loadDataset(dataSetDirectory, sparkSession)
    ds.printSchema()

    ds.show(5)
    if(dataType == TRAIN_DATA) {
      ds = ds.withColumnRenamed("TARGET", "label")
      val columns = ds
          .columns
          .filterNot(e => e == "label") ++ List("label")
      ds = ds
          .select(columns.map(col) : _*)

      ds = dropUnnecessaryColumns(ds)
    }

    if(dataType == TEST_DATA) {
      ds = ds.withColumn("label", lit(-23))
      ds = dropUnnecessaryColumns(ds)
    }

    val features = findNumericalAndCategoricalFeatures(ds)
    val categoricalFeatures = features._1
    val numericalFeatures = features._2

    val missingInfos = missingValueInfo(ds)

    val dsNoMiss = ds.na.drop()

    val numericalMissingColumns = missingInfos
      .filter(e => numericalFeatures.contains(e._1))
      .map(x => x._1)

    val categoricalMissingColumns = missingInfos
      .filter(e => categoricalFeatures.contains(e._1))
      .map(x => x._1)

    val numericalColumnsAndMeans : Map[String, Double] = numericalFeatures.map(f => {
      val average : Double = ds.select(f).agg(avg(f)).first().getDouble(0)
      ds = ds.na.fill(average, Seq(f))
      (f, average)
    }).toMap

    val categoricalColumnsAndMosts : Map[String, String] = categoricalFeatures.map(f => {
      val mode : String = dsNoMiss.groupBy(f).count().sort(col("count").desc).collect()(0)(0).toString
      ds = ds.na.fill(mode, Seq(f))
      (f, mode)
    }).toMap


//    ds = ds.withColumn("WEIGHT", weightBalance(col("label")))

    ds.coalesce(1).write
      .format("csv")
      .option("header", "true")
      .save("result/" + dataSetDirectory)

  }

  def dropColumns(inputDF: DataFrame, dropList: List[String]): DataFrame =
    dropList.foldLeft(inputDF)((df, col) => df.drop(col))

  def dropUnnecessaryColumns(frame: sql.DataFrame) : DataFrame = {
    val dataSetColumns = frame.columns
    val dropList = dataSetColumns.filter(e => e.contains("AVG") || e.contains("MODE") || e.contains("MEDI")).toList
    var newFrame = dropColumns(frame, dropList)
    newFrame
  }
  def loadDataset(dataSetPath : String, sparkSession: SparkSession): DataFrame = {

    val dataSet = sparkSession.sqlContext.read
      .format("csv")
      .option("header", true)
      .option("inferSchema", true)
      .load(dataSetPath)

    dataSet
  }

  def findNumericalAndCategoricalFeatures(dataSet : DataFrame): (Array[String], Array[String]) = {
    val categoricalColumns = dataSet.dtypes.filter(e => e._2.startsWith("String") && !e._1.startsWith("label") && !e._1.startsWith("SK_ID_CURR")).map(a => a._1)
    val numericalColumns = dataSet.dtypes.filter(e => (e._2.startsWith("Integer") || e._2.startsWith("Double")) && !e._1.startsWith("label") && !e._1.startsWith("SK_ID_CURR")).map(a => a._1)

    println("Categorical Column Size : " + categoricalColumns.size)
    println("Numerical Column Size : " + numericalColumns.size)

    (categoricalColumns, numericalColumns)
  }

  def missingValueInfo(dataSet : sql.DataFrame): Array[(String, Long)] = {
    val rowCount = dataSet.count()

    val columnNamesAndInfos = dataSet.dtypes

    val columnAndMissingValueCount = columnNamesAndInfos.map(e => {
      val columnName = e._1
      val emptyRowCount = dataSet.filter(dataSet(columnName).isNull || dataSet(columnName) === "" || dataSet(columnName).isNaN).count()
      (columnName, emptyRowCount)
    })

    val onlyMissingColumns = columnAndMissingValueCount.filter(e => e._2 != 0)
   onlyMissingColumns
  }
}
