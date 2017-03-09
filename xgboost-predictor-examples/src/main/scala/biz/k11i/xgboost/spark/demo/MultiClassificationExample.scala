package biz.k11i.xgboost.spark.demo

import biz.k11i.xgboost.spark.model.XGBoostMultiClassification
import org.apache.spark.SparkConf
import org.apache.spark.sql.SparkSession

object MultiClassificationExample {
  def main(args: Array[String]): Unit = {
    val sparkConf = new SparkConf().setAppName("MultiClassificationExample")
      .setMaster("local")
    val sparkSession = SparkSession.builder().config(sparkConf).getOrCreate()

    val modelPath = getResourcePath("/biz/k11i/xgboost/demo/model/spark/iris.model.spark")
    val testDataPath = getResourcePath("/biz/k11i/xgboost/demo/model/spark/iris.test")

    val multiclassClassifier = XGBoostMultiClassification.load(modelPath)
      .setRawPredictionCol("rawPrediction")
    val df = sparkSession.sqlContext.read.format("libsvm").load(testDataPath)

    // Predict labels
    multiclassClassifier.transform(df)
      .select("rawPrediction", "probability", "prediction", "label")
      .show(false)
  }

  def getResourcePath(name: String): String = getClass.getResource(name).getPath
}
