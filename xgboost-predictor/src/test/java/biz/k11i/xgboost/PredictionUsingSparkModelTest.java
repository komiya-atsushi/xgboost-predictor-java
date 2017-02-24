package biz.k11i.xgboost;

import biz.k11i.xgboost.util.FVec;
import org.junit.experimental.theories.DataPoints;
import org.junit.experimental.theories.Theories;
import org.junit.experimental.theories.Theory;
import org.junit.runner.RunWith;

import java.io.IOException;

@RunWith(Theories.class)
public class PredictionUsingSparkModelTest extends PredictorTest {

    static class TestDataSet {
        final String name;
        final String testDataName;


        TestDataSet(String name, String testDataName) {
            this.name = name;
            this.testDataName = testDataName;
        }
    }

    @DataPoints
    public static final TestDataSet[] dataPoints = {
            new TestDataSet("agaricus", "agaricus.txt.test.1"),
            new TestDataSet("iris", "iris.test"),
            new TestDataSet("housing", "housing.test"),
    };

    @Theory
    public void test(TestDataSet testDataSet) throws IOException {
        verifyDouble(
                "model/gbtree/spark/" + testDataSet.name + ".model.spark",
                "data/" + testDataSet.testDataName,
                "expected/gbtree/spark/" + testDataSet.name + ".predict",
                "predict",
                new PredictorFunction2<double[]>() {
                    @Override
                    public double[] predict(Predictor predictor, FVec feat) {
                        return predictor.predict(feat);
                    }
                });
    }
}
