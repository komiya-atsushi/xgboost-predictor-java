xgboost-predictor-java
======================

Pure Java implementation of [XGBoost](https://github.com/dmlc/xgboost/) predictor for online prediction tasks.


# Getting started

## Adding to dependencies

If you use **Maven**:

```xml
<repositories>
  <repository>
    <id>bintray-komiya-atsushi-maven</id>
    <url>http://dl.bintray.com/komiya-atsushi/maven</url>
  </repository>
</repository>

<dependencies>
  <dependency>
    <groupId>biz.k11i</groupId>
    <artifactId>xgboost-predictor</artifactId>
    <version>0.1.0</version>
  </dependency>
</dependencies>
```

Or **Gradle**:

```groovy
repositories {
    // Use jcenter instead of mavenCentral
    jcenter()
}

dependencies {
    compile group: 'biz.k11i', name: 'xgboost-predictor', version: '0.1.0'
}
```

## Using Predictor in Java

```java
package biz.k11i.xgboost.demo;

import biz.k11i.xgboost.Predictor;
import biz.k11i.xgboost.util.FVec;

public class HowToUseXgboostPredictor {
    public static void main(String[] args) throws java.io.IOException {

        // Load model and create Predictor
        Predictor predictor = new Predictor(
                new java.io.FileInputStream("/path/to/xgboost-model-file"));

        // Create feature vector from dense representation by array
        double[] denseArray = {0, 0, 32, 0, 0, 16, -8, 0, 0, 0};
        FVec fVecDense = FVec.Transformer.fromArray(
                denseArray,
                true /* treat zero element as N/A */);

        // Create feature vector from sparse representation by map
        FVec fVecSparse = FVec.Transformer.fromMap(
                new java.util.HashMap<Integer, Double>() {{
                    put(2, 32.);
                    put(5, 16.);
                    put(6, -8.);
                }});

        // Predict probability or classification
        double[] prediction = predictor.predict(fVecDense);

        // prediction[0] has
        //    - probability ("binary:logistic")
        //    - class label ("multi:softmax")

        // Predict leaf index of each tree
        int[] leafIndexes = predictor.predictLeaf(fVecDense);

        // leafIndexes[i] has a leaf index of i-th tree
    }
}
```


# Supported models, objective functions and API

- Models
    - "gbtree"
- Objective functions
    - "binary:logistic"
    - "binary:logitraw"
    - "multi:softmax"
    - "multi:softprob"
- API
    - Predicts probability or classification
        - `Predictor#predict(FVec)`
    - Outputs margin
        - `Predictor#predict(FVec, true /* output margin */)`
    - Predicts leaf index
        - `Predictor#predictLeaf(FVec)`
