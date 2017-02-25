package biz.k11i.xgboost;

import biz.k11i.xgboost.learner.ObjFunction;
import org.junit.experimental.theories.FromDataPoints;
import org.junit.experimental.theories.Theory;

import java.io.IOException;

public abstract class GBPredictorTestBase extends PredictionTestBase {

    static class TestDataSet {
        final String modelType;
        final String modelVersion;
        final String modelName;

        TestDataSet(String modelType, String modelVersion, String modelName) {
            this.modelType = modelType;
            this.modelVersion = modelVersion;
            this.modelName = modelName;
        }

        PredictionModel predictionModel() {
            String path = String.format("model/%s/v%s/%s.model", modelType, modelVersion, modelName);
            return new PredictionModel(path);
        }

        TestData testData() {
            return TestData.zeroBasedIndex("data/agaricus.txt.test.0");
        }

        ExpectedData expectedData(String taskName) {
            String path = String.format("model/%s/v%s/%s.%s", modelType, modelVersion, modelName, taskName);
            return new ExpectedData(path);
        }
    }

    abstract String getModelType();

    @Theory
    public void testPredict(
            @FromDataPoints("modelName") String modelName,
            @FromDataPoints("version") String version,
            boolean useJafama,
            PredictionTask predictionTask) throws IOException {

        ObjFunction.useFastMathExp(useJafama);

        TestDataSet dataSet = new TestDataSet(getModelType(), version, modelName);
        verify(
                dataSet.predictionModel(),
                dataSet.testData(),
                dataSet.expectedData(predictionTask.name),
                predictionTask);
    }

    @Theory
    public void testPredictSingle(
            @FromDataPoints("modelName") String modelName,
            @FromDataPoints("version") String version,
            boolean useJafama) throws IOException {

        if (modelName.startsWith("binary-")) {
            ObjFunction.useFastMathExp(useJafama);

            TestDataSet dataSet = new TestDataSet(getModelType(), version, modelName);
            verify(
                    dataSet.predictionModel(),
                    dataSet.testData(),
                    dataSet.expectedData("predict"),
                    PredictionTask.predictSingle());
        }
    }
}
