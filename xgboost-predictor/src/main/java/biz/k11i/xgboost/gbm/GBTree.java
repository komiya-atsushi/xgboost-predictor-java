package biz.k11i.xgboost.gbm;

import biz.k11i.xgboost.util.ModelReader;
import biz.k11i.xgboost.tree.RegTree;

import java.io.IOException;
import java.util.Map;

/**
 * Gradient boosted tree implementation.
 */
public class GBTree extends GBBase {
    private ModelParam mparam;
    private RegTree[] trees;
    private int[] tree_info;

    GBTree() {
        // do nothing
    }

    @Override
    public void loadModel(ModelReader reader, boolean with_pbuffer) throws IOException {
        mparam = new ModelParam(reader);

        trees = new RegTree[mparam.num_trees];
        for (int i = 0; i < mparam.num_trees; i++) {
            trees[i] = new RegTree();
            trees[i].loadModel(reader);
        }

        if (mparam.num_trees != 0) {
            tree_info = reader.readIntArray(mparam.num_trees);
        }

        if (mparam.num_pbuffer != 0 && with_pbuffer) {
            reader.skip(4 * mparam.predBufferSize());
            reader.skip(4 * mparam.predBufferSize());
        }
    }

    @Override
    public double[] predict(Map<Integer, Float> feat, int ntree_limit) {
        double[] preds = new double[mparam.num_output_group];
        for (int gid = 0; gid < mparam.num_output_group; gid++) {
            preds[gid] = pred(feat, gid, 0, ntree_limit);
        }
        return preds;
    }

    double pred(Map<Integer, Float> feat, int bst_group, int root_index, int ntree_limit) {
        int maxTrees = Math.min(trees.length, Math.max(trees.length - ntree_limit, 0));

        double psum = 0;
        for (int i = 0; i < maxTrees; i++) {
            if (tree_info[i] == bst_group) {
                int tid = trees[i].getLeafIndex(feat, root_index);
                psum += trees[i].leafValue(tid);
            }
        }

        return psum;
    }

    @Override
    public int[] predictLeaf(Map<Integer, Float> feat, int ntree_limit) {
        return predPath(feat, 0, ntree_limit);
    }


    int[] predPath(Map<Integer, Float> feat, int root_index, int ntree_limit) {
        int maxTrees = Math.min(trees.length, Math.max(trees.length - ntree_limit, 0));

        int[] leafIndex = new int[trees.length];
        for (int i = 0; i < maxTrees; i++) {
            leafIndex[i] = trees[i].getLeafIndex(feat, root_index);
        }
        return leafIndex;
    }


    static class ModelParam {
        /*! \brief number of trees */
        final int num_trees;
        /*! \brief number of root: default 0, means single tree */
        final int num_roots;
        /*! \brief number of features to be used by trees */
        final int num_feature;
        /*! \brief size of predicton buffer allocated used for buffering */
        final long num_pbuffer;
        /*!
         * \brief how many output group a single instance can produce
         *  this affects the behavior of number of output we have:
         *    suppose we have n instance and k group, output will be k*n
         */
        final int num_output_group;
        /*! \brief size of leaf vector needed in tree */
        final int size_leaf_vector;
        /*! \brief reserved parameters */
        final int[] reserved;

        ModelParam(ModelReader reader) throws IOException {
            num_trees = reader.readInt();
            num_roots = reader.readInt();
            num_feature = reader.readInt();
            reader.readInt(); // read padding
            num_pbuffer = reader.readLong();
            num_output_group = reader.readInt();
            size_leaf_vector = reader.readInt();
            reserved = reader.readIntArray(31);
            reader.readInt(); // read padding
        }

        long predBufferSize() {
            return num_output_group * num_pbuffer * (size_leaf_vector + 1);
        }
    }

}
