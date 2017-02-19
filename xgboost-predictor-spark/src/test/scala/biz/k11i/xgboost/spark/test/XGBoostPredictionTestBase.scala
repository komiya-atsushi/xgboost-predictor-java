package biz.k11i.xgboost.spark.test

import java.net.URL

import com.holdenkarau.spark.testing.DatasetSuiteBase
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions._
import org.scalatest.{Matchers, Suite}


trait XGBoostPredictionTestBase
  extends DatasetSuiteBase
    with Matchers {
  self: Suite =>

  def getResourcePath(name: String): String = {
    getClass.getResource(name) match {
      case null => throw new Exception(s"Resource '$name' does not exist")
      case url: URL => url.getPath
    }
  }

  def toResourceName(resourceType: String, name: String): String = {
    getResourcePath(s"/biz/k11i/xgboost/spark/$resourceType/$name")
  }

  def modelPath(name: String): String = toResourceName("model", name)

  def loadTestData(name: String, denseVector: Boolean = false): DataFrame = {
    sqlContext.read.format("libsvm")
      .option("vectorType", if (denseVector) "dense" else "sparse")
      .load(toResourceName("data", name))
  }

  def loadExpectedData(name: String): DataFrame = {
    sqlContext.read.parquet(toResourceName("expected", name))
  }

  def sizeOfVector(df: DataFrame, colName: String): Int = {
    val sizeOfVector = udf((v: Vector) => v.size)
    df.select(sizeOfVector(df(colName)).alias("vecSize"))
      .agg(max("vecSize")).collect()(0).getAs[Int](0)
  }

  def expandVectorToColumns(df: DataFrame, colName: String, vecSize: Int): DataFrame = {
    val vecToArray = udf((v: Vector) => v.toArray)
    df.select((0 until vecSize) map (i => vecToArray(df(colName))(i).alias(s"col_$i")): _*)
  }

  def assertColumnApproximateEquals(expectedDF: DataFrame, expectedColName: String,
    actualDF: DataFrame, actualColName: String, tol: Double): Unit = {
    assertColumnEquals(expectedDF, expectedColName, actualDF, actualColName, exact = false, tol)
  }

  def assertColumnExactEquals(expectedDF: DataFrame, expectedColName: String,
    actualDF: DataFrame, actualColName: String): Unit = {
    assertColumnEquals(expectedDF, expectedColName, actualDF, actualColName, exact = true)
  }

  def assertColumnEquals(expectedDF: DataFrame, expectedColName: String,
    actualDF: DataFrame, actualColName: String, exact: Boolean, tol: Double = 0.0): Unit = {

    val expectedDataType = expectedDF.schema(expectedColName).dataType
    actualDF.schema(actualColName).dataType should equal(expectedDataType)

    val dfs = expectedDataType.typeName match {
      case "vector" =>
        val vecSize = sizeOfVector(expectedDF, expectedColName)
        sizeOfVector(actualDF, actualColName) should equal(vecSize)

        (expandVectorToColumns(expectedDF, expectedColName, vecSize),
          expandVectorToColumns(actualDF, actualColName, vecSize))

      case _ =>
        (expectedDF.select(expectedDF(expectedColName).alias("col")),
          actualDF.select(actualDF(actualColName).alias("col")))
    }

    if (exact) {
      assertDataFrameEquals(dfs._1, dfs._2)
    } else {
      assertDataFrameApproximateEquals(dfs._1, dfs._2, tol)
    }
  }
}
