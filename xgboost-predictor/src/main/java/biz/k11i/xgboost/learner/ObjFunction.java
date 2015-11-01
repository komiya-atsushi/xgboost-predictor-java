package biz.k11i.xgboost.learner;

import java.util.HashMap;
import java.util.Map;

/**
 * Objective function implementations.
 */
public class ObjFunction {
    private static final Map<String, ObjFunction> FUNCTIIONS;

    static {
        ObjFunction nop = new ObjFunction();
        ObjFunction regLogistic = new RegLossObjLogistic();

        FUNCTIIONS = new HashMap<>();
        FUNCTIIONS.put("binary:logistic", regLogistic);
        FUNCTIIONS.put("binary:logitraw", nop);
        FUNCTIIONS.put("multi:softmax", new SoftmaxMultiClassObjClassify());
        FUNCTIIONS.put("multi:softprob", new SoftmaxMultiClassObjProb());
    }

    /**
     * Gets {@link ObjFunction} from given name.
     *
     * @param name name of objective function
     * @return objective function
     */
    public static ObjFunction fromName(String name) {
        ObjFunction result = FUNCTIIONS.get(name);
        if (result == null) {
            throw new IllegalArgumentException(name + " is not supported objective function.");
        }
        return result;
    }

    /**
     * Transforms prediction values.
     *
     * @param preds prediction
     */
    public double[] predTransform(double[] preds) {
        // do nothing
        return preds;
    }

    /**
     * Logistic regression.
     */
    static class RegLossObjLogistic extends ObjFunction {
        @Override
        public double[] predTransform(double[] preds) {
            for (int i = 0; i < preds.length; i++) {
                preds[i] = sigmoid(preds[i]);
            }
            return preds;
        }

        double sigmoid(double x) {
            return (1 / (1 + Math.exp(-x)));
        }
    }

    /**
     * Multiclass classification.
     */
    static class SoftmaxMultiClassObjClassify extends ObjFunction {
        @Override
        public double[] predTransform(double[] preds) {
            int maxIndex = 0;
            double max = preds[0];
            for (int i = 1; i < preds.length; i++) {
                if (max < preds[i]) {
                    maxIndex = i;
                    max = preds[i];
                }
            }

            return new double[]{maxIndex};
        }
    }

    /**
     * Multiclass classification (predicted probability).
     */
    static class SoftmaxMultiClassObjProb extends ObjFunction {
        @Override
        public double[] predTransform(double[] preds) {
            double max = preds[0];
            for (int i = 1; i < preds.length; i++) {
                max = Math.max(preds[i], max);
            }

            double sum = 0;
            for (int i = 0; i < preds.length; i++) {
                preds[i] = Math.exp(preds[i] - max);
                sum += preds[i];
            }

            for (int i = 0; i < preds.length; i++) {
                preds[i] /= (float) sum;
            }

            return preds;
        }
    }
}
