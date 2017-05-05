package biz.k11i.xgboost.spark.demo

import biz.k11i.xgboost.TemporaryFileResource
import biz.k11i.xgboost.spark.model.XGBoostMultiClassification
import org.apache.spark.SparkConf
import org.apache.spark.sql.SparkSession

object MultiClassificationExample {
  def main(args: Array[String]): Unit = {
    val sparkConf = new SparkConf().setAppName("MultiClassificationExample")
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
    val modelPath = tempFileResource.getAsPath("model/gbtree/spark/iris.model.spark").toString
    val testDataPath = tempFileResource.getAsPath("data/iris.test").toString

    val multiclassClassifier = XGBoostMultiClassification.load(modelPath)
      .setRawPredictionCol("rawPrediction")
    val df = sparkSession.sqlContext.read.format("libsvm").load(testDataPath)

    // Predict labels
    multiclassClassifier.transform(df)
      .select("rawPrediction", "probability", "prediction", "label")
      .show(false)
  }
}
