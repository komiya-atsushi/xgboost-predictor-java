package biz.k11i.xgboost;

import org.junit.Test;

import java.io.BufferedInputStream;
import java.io.IOException;
import java.io.InputStream;

public class PairwiseModelLoadingTest {
    private static final String MODEL_PATH = "model/gbtree/v47/pairwise.model";

    @Test
    public void testLoadPairwise() throws IOException {
        try (InputStream is = TestHelper.getResourceAsStream(MODEL_PATH);
             BufferedInputStream in = new BufferedInputStream(is)) {
            new Predictor(in);
        }
    }

}
