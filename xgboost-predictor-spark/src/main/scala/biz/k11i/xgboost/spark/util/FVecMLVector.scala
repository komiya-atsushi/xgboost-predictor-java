package biz.k11i.xgboost.spark.util

import biz.k11i.xgboost.util.FVec
import org.apache.spark.ml.linalg.{DenseVector, SparseVector, Vector}

object FVecMLVector {
  /**
   * Transform feature vector from spark.ml's [[Vector]] to [[FVec]].
   *
   * @param vector feature vector represented by [[Vector]]
   * @return [[FVec]] object
   */
  def transform(vector: Vector, missingValue: Double = Double.NaN): FVec = vector match {
    case dv: DenseVector =>
      if (missingValue.isNaN) {
        new FVecDenseVectorNaN(dv)
      } else {
        new FVecDenseVectorMissingValue(dv, missingValue)
      }
    case sv: SparseVector => new FVecSparseVector(sv)
  }
}

private class FVecDenseVectorNaN(dv: DenseVector)
  extends FVec {

  override def fvalue(index: Int): Double = {
    if (index >= dv.values.length) {
      Double.NaN
    } else {
      dv.values(index)
    }
  }
}

private class FVecDenseVectorMissingValue(dv: DenseVector, missingValue: Double)
  extends FVec {

  override def fvalue(index: Int): Double = {
    if (index >= dv.values.length) {
      Double.NaN
    } else {
      dv.values(index) match {
        case x if missingValue == x => Double.NaN

        case n => n
      }
    }
  }
}

private class FVecSparseVector(sv: SparseVector)
  extends FVec {
  val map: Map[Int, Double] = sv.indices.zip(sv.values).toMap

  override def fvalue(index: Int): Double = map.getOrElse(index, Double.NaN)
}
