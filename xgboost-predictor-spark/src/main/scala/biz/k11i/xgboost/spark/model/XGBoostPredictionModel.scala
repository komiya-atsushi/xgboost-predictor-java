package biz.k11i.xgboost.spark.model

import biz.k11i.xgboost.spark.SparkModelParam
import biz.k11i.xgboost.spark.util.FVecMLVector
import biz.k11i.xgboost.util.FVec
import biz.k11i.xgboost.{Predictor => XGBoostPredictor}
import org.apache.hadoop.fs.Path
import org.apache.spark.ml.PredictionModel
import org.apache.spark.ml.classification.ProbabilisticClassificationModel
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.ml.param._
import org.apache.spark.ml.util.MLReader
import org.apache.spark.mllib.linalg.VectorUDT
import org.apache.spark.sql.functions.{col, udf}
import org.apache.spark.sql.types.StructField
import org.apache.spark.sql.{DataFrame, Dataset}

private[model] trait XGBoostPredictionModel[M <: XGBoostPredictionModel[M]]
  extends PredictionModel[Vector, M] with Params {

  def setLabelCol(name: String): M = set(labelCol, name).asInstanceOf[M]

  /**
   * whether to predict leaves.
   */
  final val predictLeaves: BooleanParam = new BooleanParam(this, "predictLeaves", "whether to predict leaves")

  setDefault(predictLeaves, false)

  final def getPredictLeaves: Boolean = $(predictLeaves)

  final def setPredictLeaves(value: Boolean): M = set(predictLeaves, value).asInstanceOf[M]

  /**
   * the value treated as missing.
   */
  final val missingValue: DoubleParam = new DoubleParam(this, "missingValue", "the value treated as missing")

  setDefault(missingValue, Double.NaN)

  final def getMissingValue: Double = $(missingValue)

  final def setMissingValue(value: Double): M = set(missingValue, value).asInstanceOf[M]

  /**
   * [[XGBoostPredictor]] instance.
   */
  final val xgboostPredictor: Param[XGBoostPredictor] = new Param[XGBoostPredictor](this, "xgboostPredictor", "XGBoost Predictor")

  setDefault(xgboostPredictor, null)

  final def getXGBoostPredictor: XGBoostPredictor = $(xgboostPredictor)

  final def setXGBoostPredictor(value: XGBoostPredictor): M = set(xgboostPredictor, value).asInstanceOf[M]

  override def transform(dataset: Dataset[_]): DataFrame = {
    if (this.getPredictLeaves) {
      dataset.schema.add(StructField($(predictionCol), new VectorUDT, nullable = false))

      val predictor = this.getXGBoostPredictor
      val predictLeavesUdf = udf { (features: Vector) =>
        Vectors.dense(predictor.predictLeaf(toFVec(features)).map(_.toDouble))
      }

      dataset.withColumn($(predictionCol), predictLeavesUdf(col($(featuresCol))))

    } else {
      super.transform(dataset)
    }
  }

  override def copy(extra: ParamMap): M = defaultCopy(extra).asInstanceOf[M]

  /**
   * Generates [[FVec]] instance from [[Vector]] instance.
   *
   * @param vector feature vector represented by [[Vector]]
   * @return feature vector represented by [[FVec]]
   */
  protected def toFVec(vector: Vector): FVec = {
    FVecMLVector.transform(vector, missingValue = $(missingValue))
  }
}

private[model] trait XGBoostPrediction[M <: XGBoostPredictionModel[M]] extends MLReader[M] {
  /**
   * Load XGBoost prediction model from path in HDFS-compatible file system
   *
   * @param modelPath The path of the file representing the model
   * @return The loaded model
   */
  override def load(modelPath: String): M = {
    val path = new Path(modelPath)
    val inputStream = path.getFileSystem(sc.hadoopConfiguration).open(path)

    try {
      val xgboostPredictor = new XGBoostPredictor(inputStream)

      xgboostPredictor.getSparkModelParam match {
        case param: SparkModelParam =>
          val model = param.getModelType match {
            case "_cls_" => loadClassificationModel(xgboostPredictor, param)
            case "_reg_" => loadRegressionModel(xgboostPredictor, param)
          }

          model.setFeaturesCol(param.getFeatureCol)
            .setLabelCol(param.getLabelCol)
            .setPredictionCol(param.getPredictionCol)

        case null => newXGBoostModel(xgboostPredictor)
      }
    } catch {
      case e: Exception => throw new Exception(s"Cannot load model: $path", e)

    } finally {
      inputStream.close()
    }
  }

  private def loadClassificationModel(xgboostPredictor: XGBoostPredictor, param: SparkModelParam): M = {
    val model = newXGBoostModel(xgboostPredictor)

    model match {
      case m: ProbabilisticClassificationModel[Vector, M] =>
        m.setRawPredictionCol(param.getRawPredictionCol)
        if (param.getThresholds != null) {
          m.setThresholds(param.getThresholds)
        }
    }

    model
  }

  private def loadRegressionModel(xgboostPredictor: XGBoostPredictor, param: SparkModelParam): M = {
    newXGBoostModel(xgboostPredictor)
  }

  protected def newXGBoostModel(xgboostPredictor: XGBoostPredictor): M
}