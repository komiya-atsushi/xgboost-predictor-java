# Release notes

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
