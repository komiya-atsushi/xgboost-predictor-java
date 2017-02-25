package biz.k11i.xgboost;

import biz.k11i.xgboost.learner.ObjFunction;
import org.junit.After;
import org.junit.experimental.theories.DataPoints;
import org.junit.experimental.theories.Theories;
import org.junit.runner.RunWith;

@RunWith(Theories.class)
public class GBLinearPredictorTest extends GBPredictorTestBase {

    private static final String MODEL_TYPE = "gblinear";

    @DataPoints("modelName")
    public static final String[] MODEL_NAMES = {
            "binary-logistic",
            "binary-logitraw",
            "multi-softmax",
            "multi-softprob",
    };

    @DataPoints("version")
    public static final String[] VERSIONS = {"40", "47"};

    @DataPoints
    public static final boolean[] USE_JAFAMA = {true, false};

    @DataPoints
    public static final PredictionTask[] TASKS = {
            PredictionTask.predict(),
            PredictionTask.predictMargin()
    };

    @Override
    String getModelType() {
        return MODEL_TYPE;
    }

    @After
    public void tearDown() {
        ObjFunction.useFastMathExp(false);
    }
}
