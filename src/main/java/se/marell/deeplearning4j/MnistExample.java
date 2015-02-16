/*
 * Created by Daniel Marell 15-01-28 22:39
 */
package se.marell.deeplearning4j;

import org.apache.commons.math3.random.MersenneTwister;
import org.apache.commons.math3.random.RandomGenerator;
import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.datasets.iterator.MultipleEpochsIterator;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.models.featuredetectors.rbm.RBM;
import org.deeplearning4j.nn.api.LayerFactory;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.override.ConfOverride;
import org.deeplearning4j.nn.layers.factory.LayerFactories;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.api.activation.Activations;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Arrays;

public class MnistExample {
    private static Logger log = LoggerFactory.getLogger(MnistExample.class);

    public static void main(String[] args) throws Exception {
        RandomGenerator gen = new MersenneTwister(123);
        LayerFactory l = LayerFactories.getFactory(RBM.class);
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .weightInit(WeightInit.DISTRIBUTION)
//                .momentum(5e-1f)
                .iterations(1000)
                .render(1)
                .layerFactory(l)
                .lossFunction(LossFunctions.LossFunction.RMSE_XENT)
                .rng(gen)
                .optimizationAlgo(OptimizationAlgorithm.ITERATION_GRADIENT_DESCENT)
                .learningRate(1e-1f)
                .nIn(784)
                .nOut(10)
                .list(3)
                .hiddenLayerSizes(new int[]{500, 100})
                .override(new ConfOverride() {
                    @Override
                    public void overrideLayer(int i, NeuralNetConfiguration.Builder builder) {
                        if (i == 3) {
                            builder.weightInit(WeightInit.ZERO);
                            builder.activationFunction(Activations.softMaxRows());
                            builder.lossFunction(LossFunctions.LossFunction.MCXENT);
                        }
                    }
                })
                .build();
        System.out.println("### configured");

        MultiLayerNetwork network = new MultiLayerNetwork(conf);

        DataSetIterator dataSetIter = new MultipleEpochsIterator(1, new MnistDataSetIterator(10, 100));
        System.out.println("### calling fit");
        network.fit(dataSetIter);
        System.out.println("### fit returned");

        dataSetIter.reset();
        // Dimensionality reduced matrix
        DataSet dataSet = dataSetIter.next();
        INDArray output = network.output(dataSet.getFeatureMatrix());
        INDArray labels = dataSet.getLabels();

        System.out.println("### evaluating result");

        Evaluation eval = new Evaluation();
        eval.eval(labels, output);
        log.info(eval.stats());
        int[] predict = network.predict(dataSet.getFeatureMatrix());
        log.info("Predict " + Arrays.toString(predict));
    }
}

