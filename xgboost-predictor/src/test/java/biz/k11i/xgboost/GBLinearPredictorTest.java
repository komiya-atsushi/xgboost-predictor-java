package biz.k11i.xgboost;

import biz.k11i.xgboost.learner.ObjFunction;
import biz.k11i.xgboost.util.FVec;
import org.junit.After;
import org.junit.experimental.theories.DataPoints;
import org.junit.experimental.theories.FromDataPoints;
import org.junit.experimental.theories.Theories;
import org.junit.experimental.theories.Theory;
import org.junit.runner.RunWith;

import java.io.IOException;

@RunWith(Theories.class)
public class GBLinearPredictorTest extends PredictorTest {

    private static final String MODEL_TYPE = "gblinear";

    @DataPoints("modelName")
    public static final String[] MODEL_NAMES = {
            "binary-logistic",
            "binary-logitraw",
            "multi-softmax",
            "multi-softprob",
    };

    @DataPoints("version")
    public static final String[] VERSIONS = { "40", "47" };

    @DataPoints
    public static final boolean[] USE_JAFAMA = { true, false };

    @Theory
    public void testPredict(
            @FromDataPoints("modelName") String modelName,
            @FromDataPoints("version") String version,
            boolean useJafama) throws IOException {
        ObjFunction.useFastMathExp(useJafama);

        String path = "model/" + MODEL_TYPE + "/" + modelNameWithVersion(version, modelName) + ".model";
        final Predictor predictor = newPredictor(path);

        verifyDouble(MODEL_TYPE, modelNameWithVersion(version, modelName), "predict", new PredictorFunction<double[]>() {
            @Override
            public double[] predict(FVec feat) {
                return predictor.predict(feat);
            }
        });

        if (modelName.startsWith("binary-")) {
            // test predictSingle()
            verifyDouble(MODEL_TYPE, modelNameWithVersion(version, modelName), "predict", new PredictorFunction<double[]>() {
                @Override
                public double[] predict(FVec feat) {
                    return new double[] {predictor.predictSingle(feat)};
                }
            });
        }

        verifyDouble(MODEL_TYPE, modelNameWithVersion(version, modelName), "margin", new PredictorFunction<double[]>() {
            @Override
            public double[] predict(FVec feat) {
                return predictor.predict(feat, true);
            }
        });
    }

    @After
    public void tearDown() {
        ObjFunction.useFastMathExp(false);
    }

}