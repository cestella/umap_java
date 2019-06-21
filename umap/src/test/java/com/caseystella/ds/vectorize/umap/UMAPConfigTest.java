package com.caseystella.ds.vectorize.umap;

import org.junit.Assert;
import org.junit.Test;
import org.nd4j.linalg.primitives.Pair;

public class UMAPConfigTest {
    @Test
    public void testABComputation_default() {
        Pair<Double, Double> ab = UMAP.Builder.computeABParams(1.0, 0.1);
        Assert.assertEquals(1.5769434603113077d, ab.getKey(), 1e-5);
        Assert.assertEquals(0.8950608779109733, ab.getValue(), 1e-5);
    }
}
