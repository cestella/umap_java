package com.caseystella.ds.vectorize;

import static org.junit.Assert.assertTrue;

import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.eigen.Eigen;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

/**
 * Unit test for simple App.
 */
public class AppTest 
{
    /**
     * Rigorous Test :-)
     */
    @Test
    public void shouldAnswerWithTrue()
    {
        INDArray twoDArray = Nd4j.rand(new int[] {100, 100}).mul(100);
        INDArray eigen = Eigen.symmetricGeneralizedEigenvalues(twoDArray);
        print("Eigenvalues:", eigen);
    }
    private static void print(String tag, INDArray arr) {
        System.out.println("----------------");
        System.out.println(tag + ":\n" + arr.toString());
        System.out.println("----------------");
    }
}
