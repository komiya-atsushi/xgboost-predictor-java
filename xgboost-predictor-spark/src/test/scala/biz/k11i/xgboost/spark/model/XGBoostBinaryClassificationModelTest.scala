package biz.k11i.xgboost.spark.model

import biz.k11i.xgboost.spark.test.XGBoostPredictionTestBase
import org.scalatest.{FunSuite, Matchers}

class XGBoostBinaryClassificationModelTest
  extends FunSuite
    with XGBoostPredictionTestBase
    with Matchers {

  test("Classification using libxgboost-compatible model") {
    val model = XGBoostBinaryClassification.load(modelPath("agaricus.model"))
    val testDF = loadTestData("agaricus.txt.1.test")
    val predDF = model.transform(testDF)

    predDF.columns should contain allOf("rawPrediction", "probability", "prediction")

    val expectedDF = loadExpectedData("agaricus.predict.snappy.parquet")

    assertColumnExactEquals(
      expectedDF, "prediction",
      predDF, "prediction")

    assertColumnApproximateEquals(
      expectedDF, "probabilities",
      predDF, "probability",
      1e-7)
  }

  test("Classification using xgboost4j-spark-compatible model") {
    val model = XGBoostBinaryClassification.load(modelPath("agaricus.model.spark"))
    val testDF = loadTestData("agaricus.txt.1.test")
    val predDF = model.transform(testDF)

    predDF.columns should contain allOf("probabilities", "probability", "prediction")

    val expectedDF = loadExpectedData("agaricus.predict.snappy.parquet")

    assertColumnExactEquals(
      expectedDF, "prediction",
      predDF, "prediction")

    assertColumnApproximateEquals(
      expectedDF, "probabilities",
      predDF, "probability",
      1e-7)
  }

  test("Leaves prediction using xgboost4j-spark-compatible model") {
    val model = XGBoostBinaryClassification.load(modelPath("agaricus.model.spark"))
    val testDF = loadTestData("agaricus.txt.1.test")
    val predDF = model
      .setPredictLeaves(true)
      .transform(testDF)

    predDF.columns should contain("prediction")

    val expectedDF = loadExpectedData("agaricus.leaf.snappy.parquet")

    assertColumnExactEquals(
      expectedDF, "predLeaf",
      predDF, "prediction")
  }
}
