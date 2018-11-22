package purexgboost.xgboost.util;

import java.io.Serializable;
import java.util.Map;

/**
 * Interface of feature vector.
 */
public interface FVec extends Serializable {
    /**
     * Gets index-th value.
     *
     * @param index index
     * @return value
     */
    double fvalue(int index);

    class Transformer {
        private Transformer() {
            // do nothing
        }

        /**
         * Builds FVec from dense vector.
         *
         * @param values          float values
         * @return FVec
         */
        public static FVec fromArray(float[] values) {
            return new FVecArrayImpl.FVecFloatArrayImpl(values, true);  /* treat zero as N/A */
        }

        /**
         * Builds FVec from dense vector.
         *
         * @param values          float values
         * @param treatsValueAsNA treat specify value as N/A
         * @return FVec
         */
        public static FVec fromArray(float[] values, float treatsValueAsNA) {
            return new FVecArrayImpl.FVecFloatArrayImplement(values, treatsValueAsNA);
        }

        /**
         * Builds FVec from dense vector.
         *
         * @param values         double values
         * @return FVec
         */
        public static FVec fromArray(double[] values) {
            return new FVecArrayImpl.FVecDoubleArrayImpl(values, true);  /* treat zero as N/A */
        }

        /**
         * Builds FVec from dense vector.
         *
         * @param values          double values
         * @param treatsValueAsNA treat specify value as N/A
         * @return FVec
         */
        public static FVec fromArray(double[] values, double treatsValueAsNA) {
            return new FVecArrayImpl.FVecDoubleArrayImplement(values, treatsValueAsNA);
        }

        /**
         * Builds FVec from map.
         *
         * @param map map containing non-zero values
         * @return FVec
         */
        public static FVec fromMap(Map<Integer, ? extends Number> map) {
            return new FVecMapImpl(map);
        }
    }
}

class FVecMapImpl implements FVec {
    private final Map<Integer, ? extends Number> values;

    FVecMapImpl(Map<Integer, ? extends Number> values) {
        this.values = values;
    }

    @Override
    public double fvalue(int index) {
        Number number = values.get(index);
        if (number == null) {
            return Double.NaN;
        }

        return number.doubleValue();
    }
}

class FVecArrayImpl {
    static class FVecFloatArrayImpl implements FVec {
        private final float[] values;
        private final boolean treatsZeroAsNA;

        FVecFloatArrayImpl(float[] values, boolean treatsZeroAsNA) {
            this.values = values;
            this.treatsZeroAsNA = treatsZeroAsNA;
        }

        @Override
        public double fvalue(int index) {
            if (values.length <= index) {
                return Double.NaN;
            }

            double result = values[index];
            if (treatsZeroAsNA && result == 0) {
                return Double.NaN;
            }

            return result;
        }
    }

    static class FVecFloatArrayImplement implements FVec {
        private final float[] values;
        private final float treatsValueAsNA;

        FVecFloatArrayImplement(float[] values, float treatsValueAsNA) {
            this.values = values;
            this.treatsValueAsNA = treatsValueAsNA;
        }

        @Override
        public double fvalue(int index) {
            if (values.length <= index) {
                return Double.NaN;
            }

            double result = values[index];
            if (treatsValueAsNA == result) {
                return Double.NaN;
            }

            return result;
        }
    }

    static class FVecDoubleArrayImpl implements FVec {
        private final double[] values;
        private final boolean treatsZeroAsNA;

        FVecDoubleArrayImpl(double[] values, boolean treatsZeroAsNA) {
            this.values = values;
            this.treatsZeroAsNA = treatsZeroAsNA;
        }

        @Override
        public double fvalue(int index) {
            if (values.length <= index) {
                return Double.NaN;
            }

            double result = values[index];
            if (treatsZeroAsNA && result == 0) {
                return Double.NaN;
            }

            return values[index];
        }
    }

    static class FVecDoubleArrayImplement implements FVec {
        private final double[] values;
        private final double treatsValueAsNA;

        FVecDoubleArrayImplement(double[] values, double treatsValueAsNA) {
            this.values = values;
            this.treatsValueAsNA = treatsValueAsNA;
        }

        @Override
        public double fvalue(int index) {
            if (values.length <= index) {
                return Double.NaN;
            }

            double result = values[index];
            if (treatsValueAsNA == result) {
                return Double.NaN;
            }

            return values[index];
        }
    }
}
