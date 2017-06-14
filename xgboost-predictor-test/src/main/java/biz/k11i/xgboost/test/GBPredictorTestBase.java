package biz.k11i.xgboost.test;

import biz.k11i.xgboost.TestHelper;
import biz.k11i.xgboost.learner.ObjFunction;
import org.junit.After;
import org.junit.experimental.theories.DataPoints;
import org.junit.experimental.theories.Theory;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import static biz.k11i.xgboost.TestHelper.Expectation;
import static biz.k11i.xgboost.TestHelper.TestData;

public abstract class GBPredictorTestBase extends PredictionTestBase {

    public static class TestParameter {

        public static class Builder {
            private String modelType;
            private String[] modelNames;
            private String testDataName;
            private boolean oneBasedIndex;
            private PredictionTask[] tasks;

            public Builder modelType(String modelType) {
                this.modelType = modelType;
                return this;
            }

            public Builder modelNames(String... modelNames) {
                this.modelNames = modelNames;
                return this;
            }

            public Builder testData(String testDataName, boolean oneBasedIndex) {
                this.testDataName = testDataName;
                this.oneBasedIndex = oneBasedIndex;
                return this;
            }

            public Builder tasks(PredictionTask... tasks) {
                this.tasks = tasks;
                return this;
            }

            public List<TestParameter> build() {
                List<TestParameter> result = new ArrayList<>();

                for (String modelName : modelNames) {
                    for (PredictionTask task : tasks) {
                        TestParameter param = new TestParameter(
                                String.format("model/%s/%s.model", modelType, modelName),
                                String.format("data/%s", testDataName),
                                oneBasedIndex,
                                String.format("expectation/%s/%s.%s", modelType, modelName, task.expectationSuffix()),
                                task);
                        result.add(param);
                    }
                }

                return result;
            }

            public static List<TestParameter> merge(Builder... builders) {
                List<TestParameter> result = new ArrayList<>();

                for (Builder builder : builders) {
                    result.addAll(builder.build());
                }

                return result;
            }
        }

        private final String modelPath;
        private final String testDataPath;
        private final boolean oneBasedIndex;
        private final String expectationPath;
        private final PredictionTask task;

        TestParameter(String modelPath, String testDataPath, boolean oneBasedIndex, String expectationPath, PredictionTask task) {
            this.modelPath = modelPath;
            this.testDataPath = testDataPath;
            this.oneBasedIndex = oneBasedIndex;
            this.expectationPath = expectationPath;
            this.task = task;
        }

        public PredictionModel predictionModel() {
            return new PredictionModel(modelPath);
        }

        public TestData testData() {
            if (oneBasedIndex) {
                return TestHelper.newTestDataOfOneBasedIndex(testDataPath);
            } else {
                return TestHelper.newTestDataOfZeroBasedIndex(testDataPath);
            }
        }

        public Expectation expectation() {
            return TestHelper.newExpectation(expectationPath);
        }

        @Override
        public String toString() {
            return "{" +
                    "model: " + modelPath + ", " +
                    "testData: " + testDataPath + (oneBasedIndex ? "(one-based)" : "(zero-based)") + ", " +
                    "expectation: " + expectationPath + ", " +
                    "task: " + task.name() +
                    "}";
        }
    }

    @DataPoints
    public static final boolean[] USE_JAFAMA = {true, false};

    @After
    public void tearDown() {
        ObjFunction.useFastMathExp(false);
    }

    @Theory
    public void testPredict(
            TestParameter parameter,
            boolean useJafama) throws IOException {

        ObjFunction.useFastMathExp(useJafama);

        verify(
                parameter.predictionModel(),
                parameter.testData(),
                parameter.expectation(),
                parameter.task);
    }
}
