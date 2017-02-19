package biz.k11i.xgboost.spark.model

import biz.k11i.xgboost.spark.util.FVecMLVector
import biz.k11i.xgboost.{Predictor => XGBoostPredictor}
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.regression.RegressionModel
import org.apache.spark.ml.util.Identifiable

/**
 * XGBoost prediction model for regression task.
 *
 * @param uid               uid
 * @param _xgboostPredictor [[XGBoostPredictor]] instance
 */
class XGBoostRegressionModel(
  override val uid: String,
  _xgboostPredictor: XGBoostPredictor)
  extends RegressionModel[Vector, XGBoostRegressionModel]
    with XGBoostPredictionModel[XGBoostRegressionModel] {

  def this(xgboostPredictor: XGBoostPredictor) = this(Identifiable.randomUID("XGBoostPredictorRegressionModel"), xgboostPredictor)

  setDefault(xgboostPredictor, _xgboostPredictor)

  override protected def predict(features: Vector): Double = {
    getXGBoostPredictor.predictSingle(toFVec(features))
  }
}

object XGBoostRegression extends XGBoostPrediction[XGBoostRegressionModel] {
  override protected def newXGBoostModel(xgboostPredictor: XGBoostPredictor): XGBoostRegressionModel = new XGBoostRegressionModel(xgboostPredictor)
}
