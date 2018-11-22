# Release notes

## 0.3.1

- [#23](https://github.com/komiya-atsushi/xgboost-predictor-java/pull/23) Add method `FVec.Transformer#fromArray(float[] values, float treatsValueAsNA)`.
- [#29](https://github.com/komiya-atsushi/xgboost-predictor-java/pull/29) Support for loading reg:logistic models.
- [#35](https://github.com/komiya-atsushi/xgboost-predictor-java/pull/35) Handle ntree_limit higher than number of trees in GBTree and Dart model in graceful way.
- [#36](https://github.com/komiya-atsushi/xgboost-predictor-java/pull/36) Fixes NPE when predicting on an empty "gbtree" model.

## 0.3.0

- [#27](https://github.com/komiya-atsushi/xgboost-predictor-java/pull/27) Support DART model.

## 0.2.1

- Support an objective function: `"rank:pairwise"`

## 0.2.0

- Support XGBoost4J-Spark-generated model file format.
- Introduce [xgboost-predictor-spark](https://github.com/komiya-atsushi/xgboost-predictor-java/tree/master/xgboost-predictor-spark).


## 0.1.8

- Make `Predictor` Spark-friendly (implement `Serializable` interface, [#11](https://github.com/komiya-atsushi/xgboost-predictor-java/issues/11) )

## 0.1.7

- Support latest model file format.
    - [Commit log of xgboost](https://github.com/dmlc/xgboost/commit/0d95e863c981548b5a7ca363310fc359a9165d85#diff-53a3a623be5ce5a351a89012c7b03a31R193)

## 0.1.6

- Improve the speed performance of prediction:
    - Optimize tree retrieval performance.

## 0.1.5

- Support an objective function: `"reg:linear"`

## 0.1.4

- Improve the speed performance of prediction:
    - Introduce methods `Predictor#predictSingle()` for predicting single value efficiently.

## 0.1.3

- Improve the speed performance of prediction:
    - Use [Jafama](https://github.com/jeffhain/jafama/) for calculating sigmoid function faster.
    - Calling `ObjFunction.useFastMathExp(true)` you can use Jafama's `FastMath.exp()`. 

## 0.1.2

- #2 Add linear models (`GBLinear`).

## 0.1.1

- #1 Allow users to register their `ObjFunction`.

## 0.1.0

- Initial release.
