package biz.k11i.xgboost.demo;

import biz.k11i.xgboost.Predictor;
import biz.k11i.xgboost.util.FVec;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.util.AbstractMap.SimpleEntry;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class Example {
    public static void main(String[] args) throws IOException {
        List<SimpleEntry<Integer, FVec>> data = loadData();
        Predictor predictor = new Predictor(Example.class.getResourceAsStream("model/binary-logistic.model"));

        predictAndLogLoss(predictor, data);

        predictLeafIndex(predictor, data);
    }

    /**
     * Predicts probability and calculate its logarithmic loss using {@link Predictor#predict(FVec)}.
     *
     * @param predictor Predictor
     * @param data      test data
     */
    static void predictAndLogLoss(Predictor predictor, List<SimpleEntry<Integer, FVec>> data) {
        double sum = 0;

        for (SimpleEntry<Integer, FVec> pair : data) {

            double[] predicted = predictor.predict(pair.getValue());

            double predValue = Math.min(Math.max(predicted[0], 1e-15), 1 - 1e-15);
            int actual = pair.getKey();
            sum = actual * Math.log(predValue) + (1 - actual) * Math.log(1 - predValue);
        }

        double logLoss = -sum / data.size();

        System.out.println("Logloss: " + logLoss);
    }

    /**
     * Predicts leaf index of each tree.
     *
     * @param predictor Predictor
     * @param data test data
     */
    static void predictLeafIndex(Predictor predictor, List<SimpleEntry<Integer, FVec>> data) {
        int count = 0;
        for (SimpleEntry<Integer, FVec> pair : data) {

            int[] leafIndexes = predictor.predictLeaf(pair.getValue());

            System.out.printf("leafIndexes[%d]: %s%s",
                    count++,
                    Arrays.toString(leafIndexes),
                    System.lineSeparator());
        }
    }

    /**
     * Loads test data.
     *
     * @return test data
     */
    static List<SimpleEntry<Integer, FVec>> loadData() throws IOException {
        List<SimpleEntry<Integer, FVec>> result = new ArrayList<>();

        for (String line : Files.readAllLines(new File(Example.class.getResource("model/agaricus.txt.test").getPath()).toPath())) {
            String[] values = line.split(" ");

            Map<Integer, Float> map = new HashMap<>();

            for (int i = 1; i < values.length; i++) {
                String[] pair = values[i].split(":");
                map.put(Integer.parseInt(pair[0]), Float.parseFloat(pair[1]));
            }

            result.add(new SimpleEntry<>(Integer.parseInt(values[0]), FVec.Transformer.fromMap(map)));
        }

        return result;
    }
}
