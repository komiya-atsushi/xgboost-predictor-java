# Release notes

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
