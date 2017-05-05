package biz.k11i.xgboost;

import biz.k11i.xgboost.util.FVec;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.net.URL;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.MissingResourceException;
import java.util.concurrent.atomic.AtomicLong;

public class TestHelper {
    private static AtomicLong sequence = new AtomicLong();

    public static String getResourcePath(String name) {
        URL url = TestHelper.class.getResource(name);
        if (url == null) {
            throw new MissingResourceException(
                    String.format("[xgboost-predictor-test] not found: class = %s, name = %s",
                            TestHelper.class.getCanonicalName(), name),
                    TestHelper.class.getCanonicalName(), name);
        }

        return url.getPath();
    }

    public static InputStream getResourceAsStream(String name) {
        InputStream result = TestHelper.class.getResourceAsStream(name);
        if (result == null) {
            throw new MissingResourceException(
                    String.format("[xgboost-predictor-test] not found: class = %s, name = %s",
                            TestHelper.class.getCanonicalName(), name),
                    TestHelper.class.getCanonicalName(), name);
        }

        return result;
    }

    public static Path getResourceAsTemporaryFile(String name) {
        try (InputStream in = getResourceAsStream(name)) {
            Path tempFile = Files.createTempFile("xgboost-predictor-test-" + sequence.incrementAndGet() + "-", ".tmp");
            Files.copy(in, tempFile, StandardCopyOption.REPLACE_EXISTING);
            return tempFile;

        } catch (IOException e) {
            throw new RuntimeException("Cannot create temporary file", e);
        }
    }

    public static TestData newTestDataOfOneBasedIndex(String name) {
        return new TestData(name, true);
    }

    public static TestData newTestDataOfZeroBasedIndex(String name) {
        return new TestData(name, false);
    }

    public static Expectation newExpectation(String name) {
        return new Expectation(name);
    }

    public static class TestData {
        final String path;
        final boolean oneBasedIndex;

        TestData(String path, boolean oneBasedIndex) {
            this.path = path;
            this.oneBasedIndex = oneBasedIndex;
        }

        public List<FVec> load() throws IOException {
            try (InputStream stream = getResourceAsStream(path);
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
    }

    public static class Expectation {
        final String path;

        Expectation(String path) {
            this.path = path;
        }

        public List<double[]> load() throws IOException {
            try (InputStream stream = TestHelper.class.getResourceAsStream(path);
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
}
