package biz.k11i.xgboost;

import biz.k11i.xgboost.gbm.GradBooster;
import biz.k11i.xgboost.learner.ObjFunction;
import biz.k11i.xgboost.util.FVec;
import biz.k11i.xgboost.util.ModelReader;

import java.io.IOException;
import java.io.InputStream;

/**
 * Predicts using the Xgboost model.
 */
public class Predictor {
    private ModelParam mparam;
    private String name_obj;
    private String name_gbm;
    private ObjFunction obj;
    private GradBooster gbm;

    /**
     * Instantiates with the Xgboost model
     *
     * @param in input stream
     * @throws IOException
     */
    public Predictor(InputStream in) throws IOException {
        ModelReader reader = new ModelReader(in);

        mparam = new ModelParam(reader);
        name_obj = reader.readString();
        name_gbm = reader.readString();

        initObjGbm();

        gbm.loadModel(reader, mparam.saved_with_pbuffer != 0);
    }

    void initObjGbm() {
        obj = ObjFunction.fromName(name_obj);
        gbm = GradBooster.Factory.createGradBooster(name_gbm);
        gbm.setNumClass(mparam.num_class);
    }

    /**
     * Generates predictions for given feature vector.
     *
     * @param feat feature vector
     * @return prediction values
     */
    public double[] predict(FVec feat) {
        return predict(feat, false);
    }

    /**
     * Generates predictions for given feature vector.
     *
     * @param feat          feature vector
     * @param output_margin whether to only predict margin value instead of transformed prediction
     * @return prediction values
     */
    public double[] predict(FVec feat, boolean output_margin) {
        return predict(feat, output_margin, 0);
    }

    /**
     * Generates predictions for given feature vector.
     *
     * @param feat          feature vector
     * @param output_margin whether to only predict margin value instead of transformed prediction
     * @param ntree_limit   limit the number of trees used in prediction
     * @return prediction values
     */
    public double[] predict(FVec feat, boolean output_margin, int ntree_limit) {
        double[] preds = predictRaw(feat, ntree_limit);
        if (!output_margin) {
            return obj.predTransform(preds);
        }
        return preds;
    }

    double[] predictRaw(FVec feat, int ntree_limit) {
        double[] preds = gbm.predict(feat, ntree_limit);
        for (int i = 0; i < preds.length; i++) {
            preds[i] += mparam.base_score;
        }
        return preds;
    }

    /**
     * Predicts leaf index of each tree.
     *
     * @param feat feature vector
     * @return leaf indexes
     */
    public int[] predictLeaf(FVec feat) {
        return predictLeaf(feat, 0);
    }

    /**
     * Predicts leaf index of each tree.
     *
     * @param feat        feature vector
     * @param ntree_limit limit
     * @return leaf indexes
     */
    public int[] predictLeaf(FVec feat,
                             int ntree_limit) {
        return gbm.predictLeaf(feat, ntree_limit);
    }

    /**
     * Parameters.
     */
    static class ModelParam {
        /* \brief global bias */
        final float base_score;
        /* \brief number of features  */
        final /* unsigned */ int num_feature;
        /* \brief number of class, if it is multi-class classification  */
        final int num_class;
        /*! \brief whether the model itself is saved with pbuffer */
        final int saved_with_pbuffer;
        /*! \brief reserved field */
        final int[] reserved;

        ModelParam(ModelReader reader) throws IOException {
            base_score = reader.readFloat();
            num_feature = reader.readUnsignedInt();
            num_class = reader.readInt();
            saved_with_pbuffer = reader.readInt();
            reserved = reader.readIntArray(30);
        }
    }
}
