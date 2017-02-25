package biz.k11i.xgboost;

import biz.k11i.xgboost.util.FVec;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.closeTo;
import static org.hamcrest.Matchers.is;

public abstract class PredictionTestBase {

    static class PredictionModel {
        final String path;

        PredictionModel(String path) {
            this.path = path;
        }

        Predictor load() throws IOException {
            try (InputStream stream = PredictionTestBase.class.getResourceAsStream(path)) {
                return new Predictor(stream);
            }
        }
    }

    static class TestData {
        final String path;
        final boolean oneBasedIndex;

        private TestData(String path, boolean oneBasedIndex) {
            this.path = path;
            this.oneBasedIndex = oneBasedIndex;
        }

        List<FVec> load() throws IOException {
            try (InputStream stream = PredictionTestBase.class.getResourceAsStream(path);
                 InputStreamReader isr = new InputStreamReader(stream);
                 BufferedReader reader = new BufferedReader(isr)) {

                List<FVec> result = new ArrayList<>();
                int origin = oneBasedIndex ? 1 : 0;

                String line;
                while ((line = reader.readLine()) != null) {
                    Map<Integer, Float> feat = new HashMap<>();

                    for (String val : line.split("\\s")) {
                        if (!val.contains(":")) {
                            continue;
                        }

                        String[] pair = val.split(":");
                        feat.put(Integer.parseInt(pair[0]) - origin, Float.parseFloat(pair[1]));
                    }

                    result.add(FVec.Transformer.fromMap(feat));
                }

                return result;
            }
        }

        static TestData oneBasedIndex(String path) {
            return new TestData(path, true);
        }

        static TestData zeroBasedIndex(String path) {
            return new TestData(path, false);
        }
    }

    static class ExpectedData {
        final String path;

        ExpectedData(String path) {
            this.path = path;
        }

        List<double[]> load() throws IOException {
            try (InputStream stream = PredictionTestBase.class.getResourceAsStream(path);
                 InputStreamReader isr = new InputStreamReader(stream);
                 BufferedReader reader = new BufferedReader(isr)) {

                List<double[]> result = new ArrayList<>();

                String line;
                while ((line = reader.readLine()) != null) {
                    String[] values = line.split(",");
                    double[] valuesF = new double[values.length];
                    for (int i = 0; i < values.length; i++) {
                        valuesF[i] = Double.parseDouble(values[i]);
                    }
                    result.add(valuesF);
                }

                return result;
            }
        }
    }

    static abstract class PredictionTask {
        final String name;

        PredictionTask(String name) {
            this.name = name;
        }

        abstract double[] predict(Predictor predictor, FVec feat);

        private static double[] toDoubleArray(int[] values) {
            double[] result = new double[values.length];
            for (int i = 0; i < values.length; i++) {
                result[i] = values[i];
            }
            return result;
        }

        static PredictionTask predict() {
            return new PredictionTask("predict") {
                @Override
                double[] predict(Predictor predictor, FVec feat) {
                    return predictor.predict(feat);
                }
            };
        }

        static PredictionTask predictWithNTreeLimit(final int ntree_limit) {
            return new PredictionTask("predict_ntree") {
                @Override
                double[] predict(Predictor predictor, FVec feat) {
                    return predictor.predict(feat, false, ntree_limit);
                }
            };
        }

        static PredictionTask predictMargin() {
            return new PredictionTask("margin") {
                @Override
                double[] predict(Predictor predictor, FVec feat) {
                    return predictor.predict(feat, true);
                }
            };
        }

        static PredictionTask predictSingle() {
            return new PredictionTask("predict_single") {
                @Override
                double[] predict(Predictor predictor, FVec feat) {
                    return new double[]{predictor.predictSingle(feat)};
                }
            };
        }

        static PredictionTask predictLeaf() {
            return new PredictionTask("leaf") {
                @Override
                double[] predict(Predictor predictor, FVec feat) {
                    return toDoubleArray(predictor.predictLeaf(feat));
                }
            };
        }

        static PredictionTask predictLeafWithNTree(final int ntree_limit) {
            return new PredictionTask("leaf_ntree") {
                @Override
                double[] predict(Predictor predictor, FVec feat) {
                    return toDoubleArray(predictor.predictLeaf(feat, ntree_limit));
                }
            };
        }
    }

    void verify(
            PredictionModel model,
            TestData _testData,
            ExpectedData _expectedData,
            PredictionTask predictionTask) throws IOException {

        String context = String.format("[model: %s, test: %s, expected: %s, task: %s]",
                model.path, _testData.path, _expectedData.path, predictionTask.name);
        System.out.println(context);

        Predictor predictor = model.load();
        List<FVec> testDataList = _testData.load();
        List<double[]> expectedDataList = _expectedData.load();

        for (int i = 0; i < testDataList.size(); i++) {
            double[] predicted = predictionTask.predict(predictor, testDataList.get(i));

            assertThat(
                    String.format("result array length: %s #%d", context, i),
                    predicted.length,
                    is(expectedDataList.get(i).length));

            for (int j = 0; j < predicted.length; j++) {
                assertThat(
                        String.format("prediction value: %s #%d[%d]", context, i, j),
                        predicted[j], closeTo(expectedDataList.get(i)[j], 1e-5));
            }
        }
    }
}
