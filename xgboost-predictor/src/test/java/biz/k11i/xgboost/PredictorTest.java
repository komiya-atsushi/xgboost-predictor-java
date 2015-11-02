package biz.k11i.xgboost;

import biz.k11i.xgboost.util.FVec;
import org.junit.experimental.theories.DataPoints;
import org.junit.experimental.theories.Theories;
import org.junit.experimental.theories.Theory;
import org.junit.runner.RunWith;

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

@RunWith(Theories.class)
public class PredictorTest {
    @DataPoints
    public static final String[] MODEL_NAMES = {
            "binary-logistic",
            "binary-logitraw",
            "multi-softmax",
            "multi-softprob",
    };

    private static final List<FVec> TEST_DATA;

    static {
        try {
            TEST_DATA = loadTestData();
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    static List<FVec> loadTestData() throws IOException {
        try (InputStream stream = PredictorTest.class.getResourceAsStream("model/agaricus.txt.test");
             InputStreamReader isr = new InputStreamReader(stream);
             BufferedReader reader = new BufferedReader(isr)) {

            List<FVec> result = new ArrayList<>();

            String line;
            while ((line = reader.readLine()) != null) {
                Map<Integer, Float> feat = new HashMap<>();

                for (String val : line.split("\\s")) {
                    if (!val.contains(":")) {
                        continue;
                    }

                    String[] pair = val.split(":");
                    feat.put(Integer.parseInt(pair[0]), Float.parseFloat(pair[1]));
                }

                result.add(FVec.Transformer.fromMap(feat));
            }

            return result;
        }
    }

    static Predictor newPredictor(String resourceName) throws IOException {
        try (InputStream stream = PredictorTest.class.getResourceAsStream(resourceName)) {
            return new Predictor(stream);
        }
    }

    static List<double[]> loadExpectedData(String resourceName) throws IOException {
        try (InputStream stream = PredictorTest.class.getResourceAsStream(resourceName);
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

    interface PredictorFunction<T> {
        T predict(FVec feat);
    }

    @Theory
    public void testPredict(String modelName) throws IOException {
        final Predictor predictor = newPredictor("model/" + modelName + ".model");

        List<FVec> data = loadTestData();

        verifyDouble(modelName, "predict", new PredictorFunction<double[]>() {
            @Override
            public double[] predict(FVec feat) {
                return predictor.predict(feat);
            }
        });

        verifyDouble(modelName, "predict_ntree", new PredictorFunction<double[]>() {
            @Override
            public double[] predict(FVec feat) {
                return predictor.predict(feat, false, 1);
            }
        });

        verifyDouble(modelName, "margin", new PredictorFunction<double[]>() {
            @Override
            public double[] predict(FVec feat) {
                return predictor.predict(feat, true);
            }
        });

        verifyInt(modelName, "leaf", new PredictorFunction<int[]>() {
            @Override
            public int[] predict(FVec feat) {
                return predictor.predictLeaf(feat);
            }
        });

        verifyInt(modelName, "leaf_ntree", new PredictorFunction<int[]>() {
            @Override
            public int[] predict(FVec feat) {
                return predictor.predictLeaf(feat, 2);
            }
        });
    }

    private void verifyDouble(String modelName, String ext, PredictorFunction<double[]> function) throws IOException {
        List<double[]> expected = loadExpectedData(String.format("model/%s.%s", modelName, ext));

        for (int i = 0; i < TEST_DATA.size(); i++) {
            double[] predicted = function.predict(TEST_DATA.get(i));

            assertThat(
                    String.format("[%s.%s: %d] length", modelName, ext, i),
                    predicted.length,
                    is(expected.get(i).length));

            for (int j = 0; j < predicted.length; j++) {
                assertThat(
                        String.format("[%s.%s: %d] value[%d]", modelName, ext, i, j),
                        predicted[j], closeTo(expected.get(i)[j], 1e-5));
            }
        }
    }

    private void verifyInt(String modelName, String ext, PredictorFunction<int[]> function) throws IOException {
        List<double[]> expected = loadExpectedData(String.format("model/%s.%s", modelName, ext));

        for (int i = 0; i < TEST_DATA.size(); i++) {
            int[] predicted = function.predict(TEST_DATA.get(i));

            assertThat(
                    String.format("[%s.%s: %d] length", modelName, ext, i),
                    predicted.length,
                    is(expected.get(i).length));

            for (int j = 0; j < predicted.length; j++) {
                assertThat(
                        String.format("[%s.%s: %d] value[%d]", modelName, ext, i, j),
                        predicted[j], is((int) expected.get(i)[j]));
            }
        }
    }
}