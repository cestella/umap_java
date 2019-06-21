package com.caseystella.ds.vectorize.umap;


import com.caseystella.ds.vectorize.math.CurveFitter;
import org.apache.commons.math3.analysis.ParametricUnivariateFunction;
import org.apache.commons.math3.analysis.differentiation.DerivativeStructure;
import org.apache.commons.math3.fitting.WeightedObservedPoints;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.primitives.Pair;

import java.util.*;

/**
 * Uniform Manifold Approximation and Projection
 *     Finds a low dimensional embedding of the data that approximates
 *     an underlying manifold.
 */
public class UMAP {
   public static class Builder {

      /**
       *
       * n_neighbors: float (optional, default 15)
       *         The size of local neighborhood (in terms of number of neighboring
       *         sample points) used for manifold approximation. Larger values
       *         result in more global views of the manifold, while smaller
       *         values result in more local data being preserved. In general
       *         values should be in the range 2 to 100.
       **/
      public Builder withNumNeighbors(int numNeighbors) {
         this.numNeighbors = OptionalInt.of(numNeighbors);
         return this;
      }
      protected int getNumNeighbors() {
         return numNeighbors.getAsInt();
      }
      private OptionalInt numNeighbors = OptionalInt.of(15);

      /**
       n_components: int (optional, default 2)
       *         The dimension of the space to embed into. This defaults to 2 to
       *         provide easy visualization, but can reasonably be set to any
       *         integer value in the range 2 to 100.
       **/
      public Builder withNumComponents(int numComponents) {
         this.numComponents = OptionalInt.of(numComponents);
         return this;
      }
      protected int getNumComponents() {
         return numComponents.getAsInt();
      }
      private OptionalInt numComponents = OptionalInt.of(2);

      /*
       *     metric: string or function (optional, default 'euclidean')
       *         The metric to use to compute distances in high dimensional space.
       *         If a string is passed it must match a valid predefined metric. If
       *         a general metric is required a function that takes two 1d arrays and
       *         returns a float can be provided. For performance purposes it is
       *         required that this be a numba jit'd function. Valid string metrics
       *         include:
       *             * euclidean
       *             * manhattan
       *             * chebyshev
       *             * minkowski
       *             * canberra
       *             * braycurtis
       *             * mahalanobis
       *             * wminkowski
       *             * seuclidean
       *             * cosine
       *             * correlation
       *             * haversine
       *             * hamming
       *             * jaccard
       *             * dice
       *             * russelrao
       *             * kulsinski
       *             * rogerstanimoto
       *             * sokalmichener
       *             * sokalsneath
       *             * yule
       *         Metrics that take arguments (such as minkowski, mahalanobis etc.)
       *         can have arguments passed via the metric_kwds dictionary. At this
       *         time care must be taken and dictionary elements must be ordered
       *         appropriately; this will hopefully be fixed in the future.
       */
      public Builder withMetric(Metric metric) {
         this.metric = Optional.of(metric);
         return this;
      }
      protected Metric getMetric() {
         return metric.get();
      }
      private Optional<Metric> metric = Optional.of(Metrics.EUCLIDEAN);

      /**
       *     n_epochs: int (optional, default None)
       *         The number of training epochs to be used in optimizing the
       *         low dimensional embedding. Larger values result in more accurate
       *         embeddings. If None is specified a value will be selected based on
       *         the size of the input dataset (200 for large datasets, 500 for small).
       **/
      public Builder withNumEpochs(int numEpochs) {
         this.numEpochs= OptionalInt.of(numEpochs);
         return this;
      }
      protected int getNumEpochs() {
         return numEpochs.getAsInt();
      }
      private OptionalInt numEpochs = OptionalInt.empty();

      /**
       *     learning_rate: float (optional, default 1.0)
       *         The initial learning rate for the embedding optimization.
       */
      public Builder withLearningRate(double learningRate) {
         this.learningRate= OptionalDouble.of(learningRate);
         return this;
      }
      protected double getLearningRate() {
         return learningRate.getAsDouble();
      }
      private OptionalDouble learningRate= OptionalDouble.of(1.0);

      /**
       *     init: string (optional, default 'random')
       *         How to initialize the low dimensional embedding. Options are:
       *             * 'spectral': use a spectral embedding of the fuzzy 1-skeleton
       *             * 'random': assign initial embedding positions at random.
       *             * A numpy array of initial embedding positions.
       **/
      public Builder withInitializationStrategy(InitializationStrategy initializationStrategy) {
         this.init = Optional.of(initializationStrategy);
         return this;
      }
      protected InitializationStrategy getInitializationStrategy() {
         return init.get();
      }
      private Optional<InitializationStrategy> init = Optional.of(InitializationStrategies.RANDOM);

      /**
       *     min_dist: float (optional, default 0.1)
       *         The effective minimum distance between embedded points. Smaller values
       *         will result in a more clustered/clumped embedding where nearby points
       *         on the manifold are drawn closer together, while larger values will
       *         result on a more even dispersal of points. The value should be set
       *         relative to the ``spread`` value, which determines the scale at which
       *         embedded points will be spread out.
       **/
      public Builder withMinDistance(double minDist) {
         this.minDist = OptionalDouble.of(minDist);
         return this;
      }
      protected double getMinDistance() {
         return minDist.getAsDouble();
      }
      private OptionalDouble minDist = OptionalDouble.of(0.1);

      /**
       *     spread: float (optional, default 1.0)
       *         The effective scale of embedded points. In combination with ``min_dist``
       *         this determines how clustered/clumped the embedded points are.
       **/
      public Builder withSpread(double spread) {
         this.spread= OptionalDouble.of(spread);
         return this;
      }
      protected double getSpread() {
         return spread.getAsDouble();
      }
      private OptionalDouble spread = OptionalDouble.of(1.0);

      /**
       *     set_op_mix_ratio: float (optional, default 1.0)
       *         Interpolate between (fuzzy) union and intersection as the set operation
       *         used to combine local fuzzy simplicial sets to obtain a global fuzzy
       *         simplicial sets. Both fuzzy set operations use the product t-norm.
       *         The value of this parameter should be between 0.0 and 1.0; a value of
       *         1.0 will use a pure fuzzy union, while 0.0 will use a pure fuzzy
       *         intersection.
       **/
      public Builder withSetOpMixRatio(double setOpMixRatio) {
         this.setOpMixRatio= OptionalDouble.of(setOpMixRatio);
         return this;
      }
      protected double getSetOpMixRatio() {
         return setOpMixRatio.getAsDouble();
      }
      private OptionalDouble setOpMixRatio = OptionalDouble.of(1.0);

      /**
       *     local_connectivity: int (optional, default 1)
       *         The local connectivity required -- i.e. the number of nearest
       *         neighbors that should be assumed to be connected at a local level.
       *         The higher this value the more connected the manifold becomes
       *         locally. In practice this should be not more than the local intrinsic
       *         dimension of the manifold.
       **/
      public Builder withLocalConnectivity(int localConnectivity) {
         this.localConnectivity= OptionalInt.of(localConnectivity);
         return this;
      }
      protected int getLocalConnectivity() {
         return localConnectivity.getAsInt();
      }
      private OptionalInt localConnectivity = OptionalInt.of(1);

      /**
       *     repulsion_strength: float (optional, default 1.0)
       *         Weighting applied to negative samples in low dimensional embedding
       *         optimization. Values higher than one will result in greater weight
       *         being given to negative samples.
       **/
      public Builder withSetRepulsionStrength(double repulsionStrength) {
         this.repulsionStrength= OptionalDouble.of(repulsionStrength);
         return this;
      }
      protected double getSetRepulsionStrength() {
         return repulsionStrength.getAsDouble();
      }
      private OptionalDouble repulsionStrength = OptionalDouble.of(1.0);

      /**
       *     negative_sample_rate: int (optional, default 5)
       *         The number of negative samples to select per positive sample
       *         in the optimization process. Increasing this value will result
       *         in greater repulsive force being applied, greater optimization
       *         cost, but slightly more accuracy.
       **/
      public Builder withNegativeSampleRate(int negativeSampleRate) {
         this.negativeSampleRate= OptionalInt.of(negativeSampleRate);
         return this;
      }
      protected int getNegativeSampleRate() {
         return negativeSampleRate.getAsInt();
      }
      private OptionalInt negativeSampleRate = OptionalInt.of(5);

      /**
       *     transform_queue_size: float (optional, default 4.0)
       *         For transform operations (embedding new points using a trained model_
       *         this will control how aggressively to search for nearest neighbors.
       *         Larger values will result in slower performance but more accurate
       *         nearest neighbor evaluation.
       **/
      public Builder withTransformQueueSize(double transformQueueSize) {
         this.transformQueueSize= OptionalDouble.of(transformQueueSize);
         return this;
      }
      protected double getTransformQueueSize() {
         return transformQueueSize.getAsDouble();
      }
      private OptionalDouble transformQueueSize = OptionalDouble.of(4.0);

      /**
       *     a: float (optional, default None)
       *         More specific parameters controlling the embedding. If None these
       *         values are set automatically as determined by ``min_dist`` and
       *         ``spread``.
       **/
      public Builder withA(double a) {
         this.a = OptionalDouble.of(a);
         return this;
      }
      protected double getA() {
         if(a.isPresent()) {
            return a.getAsDouble();
         }
         throw new UnsupportedOperationException("A needs to be implemented");
      }
      private OptionalDouble a = OptionalDouble.empty();

      /**
       *     b: float (optional, default None)
       *         More specific parameters controlling the embedding. If None these
       *         values are set automatically as determined by ``min_dist`` and
       *         ``spread``.
       **/
      public Builder withB(double b) {
         this.b = OptionalDouble.of(b);
         return this;
      }
      protected double getB() {
         if(b.isPresent()) {
            return b.getAsDouble();
         }
         throw new UnsupportedOperationException("A needs to be implemented");
      }
      private OptionalDouble b = OptionalDouble.empty();

      /**
       *     random_state: int, RandomState instance or None, optional (default: None)
       *         If int, random_state is the seed used by the random number generator;
       *         If RandomState instance, random_state is the random number generator;
       *         If None, the random number generator is the RandomState instance used
       *         by `np.random`.
       **/
      public Builder withRandomState(long seed) {
         this.randomState = OptionalLong.of(seed);
         return this;
      }
      protected long getRandomState() {
         return randomState.getAsLong();
      }
      private OptionalLong randomState = OptionalLong.of(System.currentTimeMillis());

      /**
       *     metric_kwds: dict (optional, default None)
       *         Arguments to pass on to the metric, such as the ``p`` value for
       *         Minkowski distance. If None then no arguments are passed on.
       **/
      public Builder withMetricConfig(Map<String, Object> args) {
         this.metricConfig = Optional.of(args);
         return this;
      }
      protected Map<String, Object> getMetricConfig() {
         return metricConfig.get();
      }
      private Optional<Map<String, Object>> metricConfig = Optional.of(new HashMap<>());

      /**
       *     angular_rp_forest: bool (optional, default False)
       *         Whether to use an angular random projection forest to initialise
       *         the approximate nearest neighbor search. This can be faster, but is
       *         mostly on useful for metric that use an angular style distance such
       *         as cosine, correlation etc. In the case of those metrics angular forests
       *         will be chosen automatically.
       **/
      public Builder withAngularRPForest(boolean b) {
         this.useAngularRPForest = Optional.of(b);
         return this;
      }
      protected boolean useAngularRPForest() {
         return useAngularRPForest.get();
      }
      private Optional<Boolean> useAngularRPForest = Optional.of(false);
      /**
       *     target_n_neighbors: int (optional, default -1)
       *         The number of nearest neighbors to use to construct the target simplcial
       *         set. If set to -1 use the ``n_neighbors`` value.
       **/
      public Builder withTargetNumNeighbors(int targetNumNeighbors) {
         this.targetNumNeighbors= OptionalInt.of(targetNumNeighbors);
         return this;
      }
      protected int getTargetNumNeighbors() {
         return targetNumNeighbors.getAsInt();
      }
      private OptionalInt targetNumNeighbors = OptionalInt.of(-1);

      /**
       *     target_metric: string or callable (optional, default 'categorical')
       *         The metric used to measure distance for a target array is using supervised
       *         dimension reduction. By default this is 'categorical' which will measure
       *         distance in terms of whether categories match or are different. Furthermore,
       *         if semi-supervised is required target values of -1 will be trated as
       *         unlabelled under the 'categorical' metric. If the target array takes
       *         continuous values (e.g. for a regression problem) then metric of 'l1'
       *         or 'l2' is probably more appropriate.
       **/
      public Builder withTargetMetric(TargetMetric metric) {
         this.targetMetric = Optional.of(metric);
         return this;
      }
      protected TargetMetric getTargetMetric() {
         return targetMetric.get();
      }
      private Optional<TargetMetric> targetMetric = Optional.of(TargetMetrics.CATEGORICAL);

      /**
       *     target_metric_kwds: dict (optional, default None)
       *         Keyword argument to pass to the target metric when performing
       *         supervised dimension reduction. If None then no arguments are passed on.
       **/
      public Builder withTargetMetricConfig(Map<String, Object> args) {
         this.targetMetricConfig = Optional.of(args);
         return this;
      }
      protected Map<String, Object> getTargetMetricConfig() {
         return targetMetricConfig.get();
      }
      private Optional<Map<String, Object>> targetMetricConfig = Optional.of(new HashMap<>());

      /**
       *     target_weight: float (optional, default 0.5)
       *         weighting factor between data topology and target topology. A value of
       *         0.0 weights entirely on data, a value of 1.0 weights entirely on target.
       *         The default of 0.5 balances the weighting equally between data and target.
       **/
      public Builder withTargetWeight(double targetWeight) {
         this.targetWeight = OptionalDouble.of(targetWeight);
         return this;
      }
      protected double getTargetWeight() {
         return targetWeight.getAsDouble();
      }
      private OptionalDouble targetWeight = OptionalDouble.of(0.5);

      /**
       *     transform_seed: int (optional, default 42)
       *         Random seed used for the stochastic aspects of the transform operation.
       *         This ensures consistency in transform operations.
       **/
      public Builder withTransformSeed(long seed) {
         this.transformSeed = OptionalLong.of(seed);
         return this;
      }
      protected long getTransformSeed() {
         return transformSeed.getAsLong();
      }
      private OptionalLong transformSeed = OptionalLong.of(System.currentTimeMillis());

      /**
       *     verbose: bool (optional, default False)
       *         Controls verbosity of logging.
       **/
      public Builder withVerbose(boolean b) {
         this.verbose = Optional.of(b);
         return this;
      }
      protected boolean verbose() {
         return verbose.get();
      }
      private Optional<Boolean> verbose = Optional.of(false);

      public UMAP build() {
         validate();
         if(!a.isPresent() && !b.isPresent()) {
            Pair<Double, Double> ab = computeABParams(getSpread(), getMinDistance());
            a = OptionalDouble.of(ab.getKey());
            b = OptionalDouble.of(ab.getValue());
         }
         return new UMAP(this);
      }

      protected static Pair<Double, Double> computeABParams(double spread, double minDist) {
         /*
          *    """Fit a, b params for the differentiable curve used in lower
          *     dimensional fuzzy simplicial complex construction. We want the
          *     smooth curve (from a pre-defined family with simple gradient) that
          *     best matches an offset exponential decay.
          *     """
          *
          *     def curve(x, a, b):
          *         return 1.0 / (1.0 + a * x ** (2 * b))
          *
          *     xv = np.linspace(0, spread * 3, 300)
          *     yv = np.zeros(xv.shape)
          *     yv[xv < min_dist] = 1.0
          *     yv[xv >= min_dist] = np.exp(-(xv[xv >= min_dist] - min_dist) / spread)
          *     params, covar = curve_fit(curve, xv, yv)
          *     return params[0], params[1]
          */
         final WeightedObservedPoints obs = new WeightedObservedPoints();
         for(double x : Nd4j.linspace(0, spread * 3, 300, DataType.DOUBLE).toDoubleVector()) {
            double y = 1.0d;
            if(x >= minDist) {
               y = Math.exp(-1*(x - minDist) / spread);
            }
            obs.add(x, y);
         }
         ParametricUnivariateFunction func = new ParametricUnivariateFunction() {
            @Override
            public double value(double x, double... parameters) {
               return 1.0 / (1.0 + parameters[0]* Math.pow(x, 2*parameters[1]));
            }

            @Override
            public double[] gradient(double x, double... parameters) {
               final double a = parameters[0];
               final double b = parameters[1];
               DerivativeStructure one = new DerivativeStructure(2, 1, 1.0);
               DerivativeStructure two = new DerivativeStructure(2, 1, 2.0);
               DerivativeStructure aDev = new DerivativeStructure(2, 1, 0, a);
               DerivativeStructure bDev = new DerivativeStructure(2, 1, 1, b);
               DerivativeStructure y = one.divide(one.add( aDev.multiply(DerivativeStructure.pow(x, two.multiply(bDev)))));

               return new double[] {
                       y.getPartialDerivative(1, 0),
                       y.getPartialDerivative(0, 1)
               };
            }
         };
         double[] params = CurveFitter.fit(func, obs.toList(), new double[]{2.0, 2.0});

         return Pair.of(params[0], params[1]);
      }

      private void validate() {
         boolean invalidAbCondition = (a.isPresent() && !b.isPresent()) || (b.isPresent() && !a.isPresent());
         if(invalidAbCondition) {
            throw new IllegalStateException("You must either specify both a and b or neither.");
         }
      }
   }

   Builder config;
   private UMAP(Builder config) {
      this.config = config;
   }
}
