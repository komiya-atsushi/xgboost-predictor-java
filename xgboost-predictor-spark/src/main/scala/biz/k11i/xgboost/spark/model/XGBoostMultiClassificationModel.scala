package biz.k11i.xgboost.spark.model

import biz.k11i.xgboost.spark.util.FVecMLVector
import biz.k11i.xgboost.{Predictor => XGBoostPredictor}
import org.apache.spark.ml.classification.ProbabilisticClassificationModel
import org.apache.spark.ml.linalg.{DenseVector, SparseVector, Vector, Vectors}
import org.apache.spark.ml.util.Identifiable

/**
 * XGBoost prediction model for multiclass classification task.
 *
 * @param uid               uid
 * @param _xgboostPredictor [[XGBoostPredictor]] instance
 */
class XGBoostMultiClassificationModel(
  override val uid: String,
  _xgboostPredictor: XGBoostPredictor)
  extends ProbabilisticClassificationModel[Vector, XGBoostMultiClassificationModel]
    with XGBoostPredictionModel[XGBoostMultiClassificationModel] {

  def this(xgboostPredictor: XGBoostPredictor) = this(Identifiable.randomUID("XGBoostPredictorMultiClassificationModel"), xgboostPredictor)

  setDefault(xgboostPredictor, _xgboostPredictor)

  override def numClasses: Int = getXGBoostPredictor.getNumClass

  override protected def predictRaw(features: Vector): Vector = {
    val predictions = getXGBoostPredictor.predict(toFVec(features), true)
    Vectors.dense(predictions)
  }

  override protected def raw2probabilityInPlace(rawPrediction: Vector): Vector = {
    rawPrediction match {
      case dv: DenseVector =>
        val max = dv.values.max
        val values = dv.values.map(v => Math.exp(v - max)).array
        val sum = values.sum
        Vectors.dense(values.map(_ / sum).array)

      case _: SparseVector =>
        throw new Exception("rawPrediction should be DenseVector")
    }
  }

  override protected def raw2probability(rawPrediction: Vector): Vector = raw2probabilityInPlace(rawPrediction)
}

object XGBoostMultiClassification extends XGBoostPrediction[XGBoostMultiClassificationModel] {
  override protected def newXGBoostModel(xgboostPredictor: XGBoostPredictor): XGBoostMultiClassificationModel = {
    new XGBoostMultiClassificationModel(xgboostPredictor)
  }
}