package biz.k11i.xgboost.gbm;

import biz.k11i.xgboost.tree.RegTree;
import biz.k11i.xgboost.util.FVec;
import biz.k11i.xgboost.util.ModelReader;

import java.io.IOException;

/**
 * Gradient boosted DART tree implementation.
 */
public class Dart extends GBTree {
    private float[] weightDrop;

    Dart() {
        // do nothing
    }

    @Override
    public void loadModel(ModelReader reader, boolean with_pbuffer) throws IOException {
        super.loadModel(reader, with_pbuffer);
        if (mparam.num_trees != 0) {
            long size = reader.readLong();
            assert mparam.num_trees == size;
            weightDrop = reader.readFloatArray((int)size);
        }
    }

    double pred(FVec feat, int bst_group, int root_index, int ntree_limit) {
        RegTree[] trees = _groupTrees[bst_group];
        int treeleft = ntree_limit == 0 ? trees.length : ntree_limit;

        double psum = 0;
        for (int i = 0; i < treeleft; i++) {
            psum += weightDrop[i] * trees[i].getLeafValue(feat, root_index);
        }

        return psum;
    }
}
