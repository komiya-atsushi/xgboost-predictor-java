package biz.k11i.xgboost.spark.model

import biz.k11i.xgboost.spark.test.XGBoostPredictionTestBase
import org.scalatest.{FunSuite, Matchers}

class XGBoostRegressionModelTest
  extends FunSuite
    with XGBoostPredictionTestBase
    with Matchers {

  test("Regression using libxgboost-compatible model") {
    val model = XGBoostRegression.load(modelPath("housing.model"))
    val testDF = loadTestData("housing.test", denseVector = true)
    val predDF = model.transform(testDF)

    predDF.columns should contain("prediction")

    val expectedDF = loadExpectedData("housing.predict.snappy.parquet")

    assertColumnApproximateEquals(
      expectedDF, "prediction",
      predDF, "prediction",
      1e-5)
  }

  test("Regression using xgboost4j-spark-compatible model") {
    val model = XGBoostRegression.load(modelPath("housing.model.spark"))
    val testDF = loadTestData("housing.test", denseVector = true)
    val predDF = model.transform(testDF)

    predDF.columns should contain("prediction")

    val expectedDF = loadExpectedData("housing.predict.snappy.parquet")

    assertColumnApproximateEquals(
      expectedDF, "prediction",
      predDF, "prediction",
      1e-5)
  }

  test("Leaves prediction using xgboost4j-spark-compatible model") {
    val model = XGBoostRegression.load(modelPath("housing.model.spark"))
    val testDF = loadTestData("housing.test", denseVector = true)
    val predDF = model
      .setPredictLeaves(true)
      .transform(testDF)

    predDF.columns should contain("prediction")

    val expectedDF = loadExpectedData("housing.leaf.snappy.parquet")

    assertColumnExactEquals(
      expectedDF, "predLeaf",
      predDF, "prediction")
  }
}
