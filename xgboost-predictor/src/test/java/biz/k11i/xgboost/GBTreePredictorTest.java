package biz.k11i.xgboost;

import biz.k11i.xgboost.config.PredictorConfiguration;
import biz.k11i.xgboost.learner.ObjFunction;
import biz.k11i.xgboost.test.GBPredictorTestBase;
import biz.k11i.xgboost.test.TestParameters;
import org.junit.After;
import org.junit.experimental.theories.DataPoint;
import org.junit.experimental.theories.DataPoints;
import org.junit.experimental.theories.Theories;
import org.junit.runner.RunWith;

import java.util.List;

@RunWith(Theories.class)
public class GBTreePredictorTest extends GBPredictorTestBase {
    @DataPoints
    public static final List<TestParameter> PARAMETERS = TestParameters.testParametersForGBTree();

    @DataPoints
    public static final boolean[] USE_JAFAMA = {true, false};

    @DataPoint
    public static final PredictorConfiguration CONFIGURATION = PredictorConfiguration.DEFAULT;

    @After
    public void tearDown() {
        ObjFunction.useFastMathExp(false);
    }
}
