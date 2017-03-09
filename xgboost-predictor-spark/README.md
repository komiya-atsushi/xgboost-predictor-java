xgboost-predictor-spark
=======================

Apache Spark integration of [xgboost-predictor](https://github.com/komiya-atsushi/xgboost-predictor-java).

# Features

- *Pure Java / Scala implementation*
    - Runs on Spark without any native dependencies.
- *DataFrame-friendly API*
    - Provides binary classification, multi-class classification and regression functions as Spark ML API.
- *Support both C-based and XGBoost4J-Spark generated model format*
    - Automatically detects format of model and loads as `XGBoostPredictionModel` instance.


# Requirements

- Apache Spark 2.0.0 or higher version


# Getting started

## Add dependencies to build.sbt

```scala
// Add Bintray repository
resolvers += Resolver.bintrayRepo("komiya-atsushi", "maven")

libraryDependencies ++= Seq(
  // xgboost-predictor-spark requires Apache Spark version 2.0.0 or higher
  "org.apache.spark" %% "spark-mllib"             % "2.0.2" % "provided",
  "biz.k11i"         %% "xgboost-predictor-spark" % "0.2.0"
)
```

## Spark application example

```scala
object XGBoostPredictorSparkExample {
  def main(args: Array[String]): Unit = {
    val sparkConf = new SparkConf().setAppName("XGBoostPredictorSparkExample")
    val sparkSession = SparkSession.builder().config(sparkConf).getOrCreate()

    val df = sparkSession.sqlContext.read.format("libsvm").load("/path/to/test/data")

    // Load model as XGBoostBinaryClassificationModel instance
    val binaryClassifier = XGBoostBinaryClassification.load("/path/to/model")

    // Predict labels of test data
    val predDF = binaryClassifier.transform(df)

    // Prediction results are stored into the column "prediction" by default 
    predDF.select("prediction").show()

    // Predict leaves
    binaryClassifier
      .setPredictLeaves(true)
      .transform(df)
      .select("prediction")
      .show()

    // How to specify missing value explicitly
    // (This works only dense feature vector)
    binaryClassifier
      .setMissingValue(-1.0)
      .transform(denseVectorDF)
      .select("prediction")
      .show()
  }
}
```

See also [xgboost-predictor-examples](https://github.com/komiya-atsushi/xgboost-predictor-java/tree/master/xgboost-predictor-examples/src/main/scala/biz/k11i/xgboost/spark/demo).


# API

## Binary classification

- `XGBoostBinaryClassification.load()`
    - Loads binary classification model ("binary:logistic").
- `XGBoostBinaryClassificationModel`
    - Model class for binary classification.
    - This class extends [`ProbabilisticClassificationModel`](https://spark.apache.org/docs/latest/api/scala/index.html#org.apache.spark.ml.classification.ProbabilisticClassificationModel)


## Multi-class classification

- `XGBoostMultiClassification.load()`
    - Loads multi-class classification model ("multi:softmax")
- `XGBoostMultiClassificationModel`
    - Model class for multi-class classification
    - This class extends [`ProbabilisticClassificationModel`](https://spark.apache.org/docs/latest/api/scala/index.html#org.apache.spark.ml.classification.ProbabilisticClassificationModel)


## Regression

- `XGBoostRegression.load()`
    - Loads regression model ("reg:linear")
- `XGBoostRegressionModel`
    - Model class for regression
    - This class extends [`RegressionModel`](https://spark.apache.org/docs/latest/api/scala/index.html#org.apache.spark.ml.regression.RegressionModel)
