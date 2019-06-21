package com.caseystella.ds.vectorize.math;

import org.apache.commons.math3.analysis.ParametricUnivariateFunction;
import org.apache.commons.math3.fitting.AbstractCurveFitter;
import org.apache.commons.math3.fitting.WeightedObservedPoint;
import org.apache.commons.math3.fitting.leastsquares.LeastSquaresBuilder;
import org.apache.commons.math3.fitting.leastsquares.LeastSquaresProblem;
import org.apache.commons.math3.linear.DiagonalMatrix;

import java.util.Collection;

public class CurveFitter {
    public static double[] fit(ParametricUnivariateFunction func, Collection<WeightedObservedPoint> points, double[] initialGuess) {
        AbstractCurveFitter fitter = new AbstractCurveFitter() {
            @Override
            protected LeastSquaresProblem getProblem(Collection<WeightedObservedPoint> points) {
                final int len = points.size();
                final double[] target  = new double[len];
                final double[] weights = new double[len];

                int i = 0;
                for(WeightedObservedPoint point : points) {
                    target[i]  = point.getY();
                    weights[i] = point.getWeight();
                    i += 1;
                }

                final AbstractCurveFitter.TheoreticalValuesFunction model = new
                        AbstractCurveFitter.TheoreticalValuesFunction(func, points);

                return new LeastSquaresBuilder().
                        maxEvaluations(Integer.MAX_VALUE).
                        maxIterations(Integer.MAX_VALUE).
                        start(initialGuess).
                        target(target).
                        weight(new DiagonalMatrix(weights)).
                        model(model.getModelFunction(), model.getModelFunctionJacobian()).
                        build();
            }
        };
        return fitter.fit(points);
    }
}
