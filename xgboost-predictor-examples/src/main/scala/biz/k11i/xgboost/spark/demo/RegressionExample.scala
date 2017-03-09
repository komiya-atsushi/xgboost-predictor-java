package biz.k11i.xgboost.spark.demo

import biz.k11i.xgboost.spark.model.XGBoostRegression
import org.apache.spark.SparkConf
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.sql.SparkSession

object RegressionExample {
  def main(args: Array[String]): Unit = {
    val sparkConf = new SparkConf().setAppName("RegressionExample")
      .setMaster("local")
    val sparkSession = SparkSession.builder().config(sparkConf).getOrCreate()

    val modelPath = getResourcePath("/biz/k11i/xgboost/demo/model/spark/housing.model.spark")
    val testDataPath = getResourcePath("/biz/k11i/xgboost/demo/model/spark/housing.test")

    val regressor = XGBoostRegression.load(modelPath)
    val df = sparkSession.sqlContext.read
      .format("libsvm")
      .option("vectorType", "dense")
      .load(testDataPath)

    // Predict prices
    val predDF = regressor.transform(df)
    predDF.select("prediction", "label")
      .show()

    // Evaluate
    val rmse = new RegressionEvaluator()
      .setMetricName("rmse")
      .evaluate(predDF)

    println(s"RMSE: $rmse")
  }

  def getResourcePath(name: String): String = getClass.getResource(name).getPath
}
