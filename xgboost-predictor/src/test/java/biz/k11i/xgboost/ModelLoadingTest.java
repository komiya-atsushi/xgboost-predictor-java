package biz.k11i.xgboost;

import org.junit.Test;

import java.io.BufferedInputStream;
import java.io.IOException;
import java.io.InputStream;

public class ModelLoadingTest {
    private static final String MODEL_PATH = "model/gbtree/v40/binary-logistic.model";

    @Test
    public void testBuffered() throws IOException {
        try (InputStream is = ModelLoadingTest.class.getResourceAsStream(MODEL_PATH);
             BufferedInputStream in = new BufferedInputStream(is)) {
            new Predictor(in);
        }
    }

    @Test
    public void testLazy() throws IOException {
        try (InputStream is = ModelLoadingTest.class.getResourceAsStream(MODEL_PATH);
             LazyInputStream in = new LazyInputStream(is)) {
            new Predictor(in);
        }
    }

    static class LazyInputStream extends InputStream {
        private final InputStream in;

        LazyInputStream(InputStream in) {
            this.in = in;
        }

        @Override
        public int read() throws IOException {
            return in.read();
        }

        // InputStream#read(byte[], int, int) doesn't guarantee that the method fills the buffer fully.
        @Override
        public int read(byte[] b, int off, int len) throws IOException {
            if (b == null) {
                throw new NullPointerException();
            } else if (off < 0 || len < 0 || len > b.length - off) {
                throw new IndexOutOfBoundsException();
            } else if (len == 0) {
                return 0;
            }

            int c = read();
            b[off] = (byte) c;

            return 1;
        }
    }
}
