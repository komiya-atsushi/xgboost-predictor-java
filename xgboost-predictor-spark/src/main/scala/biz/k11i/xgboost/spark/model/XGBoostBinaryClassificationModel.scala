package biz.k11i.xgboost.spark.model

import biz.k11i.xgboost.spark.util.FVecMLVector
import biz.k11i.xgboost.{Predictor => XGBoostPredictor}
import org.apache.spark.ml.classification.ProbabilisticClassificationModel
import org.apache.spark.ml.linalg.{DenseVector, SparseVector, Vector, Vectors}
import org.apache.spark.ml.util.Identifiable

/**
 * XGBoost prediction model for binary classification task.
 *
 * @param uid               uid
 * @param _xgboostPredictor [[XGBoostPredictor]] instance
 */
class XGBoostBinaryClassificationModel(
  override val uid: String,
  _xgboostPredictor: XGBoostPredictor)
  extends ProbabilisticClassificationModel[Vector, XGBoostBinaryClassificationModel]
    with XGBoostPredictionModel[XGBoostBinaryClassificationModel] {

  def this(xgboostPredictor: XGBoostPredictor) = this(Identifiable.randomUID("XGBoostPredictorBinaryClassificationModel"), xgboostPredictor)

  setDefault(xgboostPredictor, _xgboostPredictor)

  override def numClasses: Int = 2

  override protected def predictRaw(features: Vector): Vector = {
    val pred = getXGBoostPredictor.predictSingle(toFVec(features), true)
    Vectors.dense(Array(-pred, pred))
  }

  override protected def raw2probabilityInPlace(rawPrediction: Vector): Vector = {
    rawPrediction match {
      case dv: DenseVector =>
        Vectors.dense(dv.values.map(v => 1 / (1 + Math.exp(-v))).array)

      case _: SparseVector =>
        throw new Exception("rawPrediction should be DenseVector")
    }
  }

  override protected def raw2probability(rawPrediction: Vector): Vector = raw2probabilityInPlace(rawPrediction)
}

object XGBoostBinaryClassification extends XGBoostPrediction[XGBoostBinaryClassificationModel] {
  override protected def newXGBoostModel(xgboostPredictor: XGBoostPredictor): XGBoostBinaryClassificationModel = {
    new XGBoostBinaryClassificationModel(xgboostPredictor)
  }
}