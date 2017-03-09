package biz.k11i.xgboost.spark.demo

import biz.k11i.xgboost.spark.model.XGBoostBinaryClassification
import org.apache.spark.SparkConf
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.sql.SparkSession

object BinaryClassificationExample {
  def main(args: Array[String]): Unit = {
    val sparkConf = new SparkConf().setAppName("BinaryClassificationExample")
      .setMaster("local")
    val sparkSession = SparkSession.builder().config(sparkConf).getOrCreate()

    val modelPath = getResourcePath("/biz/k11i/xgboost/demo/model/spark/agaricus.model.spark")
    val testDataPath = getResourcePath("/biz/k11i/xgboost/demo/model/spark/agaricus.txt.test")

    val binaryClassifier = XGBoostBinaryClassification.load(modelPath)
      .setRawPredictionCol("rawPrediction")
    val df = sparkSession.sqlContext.read.format("libsvm").load(testDataPath)

    // Predict labels
    val predDF = binaryClassifier.transform(df)
    predDF
      .select("rawPrediction", "probability", "prediction", "label")
      .show(false)

    // Predict leaves
    binaryClassifier.setPredictLeaves(true)
      .transform(df)
      .show(false)

    // Evaluate
    val areaUnderROC = new BinaryClassificationEvaluator()
      .setMetricName("areaUnderROC")
      .evaluate(predDF)

    println(s"AUC: $areaUnderROC")
  }

  def getResourcePath(name: String): String = getClass.getResource(name).getPath
}
