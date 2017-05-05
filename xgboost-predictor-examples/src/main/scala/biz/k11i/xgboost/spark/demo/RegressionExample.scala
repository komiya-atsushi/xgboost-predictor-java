package biz.k11i.xgboost.spark.demo

import biz.k11i.xgboost.TemporaryFileResource
import biz.k11i.xgboost.spark.model.XGBoostRegression
import org.apache.spark.SparkConf
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.sql.SparkSession

object RegressionExample {
  def main(args: Array[String]): Unit = {
    val sparkConf = new SparkConf().setAppName("RegressionExample")
      .setMaster("local")
    val sparkSession = SparkSession.builder().config(sparkConf).getOrCreate()
    val tempFileResource = new TemporaryFileResource

    try {
      run(sparkSession, tempFileResource)
    } finally {
      tempFileResource.close()
    }
  }

  def run(sparkSession: SparkSession, tempFileResource: TemporaryFileResource) {
    val modelPath = tempFileResource.getAsPath("model/gbtree/spark/housing.model.spark").toString
    val testDataPath = tempFileResource.getAsPath("data/housing.test").toString

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
}
