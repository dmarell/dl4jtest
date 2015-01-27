package se.marell.deeplearning4j;

import org.apache.commons.math3.random.MersenneTwister;
import org.apache.commons.math3.random.RandomGenerator;
import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.IrisDataSetIterator;
import org.deeplearning4j.distributions.Distributions;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.models.classifiers.dbn.DBN;
import org.deeplearning4j.models.featuredetectors.rbm.RBM;
import org.deeplearning4j.nn.WeightInit;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.nd4j.linalg.api.activation.Activations;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Arrays;

/**
 * Created by agibsonccc on 9/12/14.
 */
public class IrisExample {
    private static Logger log = LoggerFactory.getLogger(IrisExample.class);

    public static void main(String[] args) {
        RandomGenerator gen = new MersenneTwister(123);

        NeuralNetConfiguration conf = new NeuralNetConfiguration.Builder()
                .iterations(100)
                .weightInit(WeightInit.DISTRIBUTION)
                .dist(Distributions.normal(gen, 1e-3))
                .constrainGradientToUnitNorm(false)
                .lossFunction(LossFunctions.LossFunction.RECONSTRUCTION_CROSSENTROPY)
                .activationFunction(Activations.tanh())
                .rng(gen).visibleUnit(RBM.VisibleUnit.GAUSSIAN)
                .hiddenUnit(RBM.HiddenUnit.RECTIFIED)
                .dropOut(0.3f)
                .learningRate(1e-3f)
                .nIn(4)
                .nOut(3)
                .build();

        DBN d = new DBN.Builder()
                .configure(conf)
                .hiddenLayerSizes(new int[]{3})
                .build();

        NeuralNetConfiguration.setClassifier(d.getOutputLayer().conf());

        DataSetIterator iter = new IrisDataSetIterator(150, 150);

        //fetch first
        DataSet next = iter.next(110);
        next.normalizeZeroMeanZeroUnitVariance();

        d.fit(next);

        Evaluation eval = new Evaluation();
        INDArray output = d.output(next.getFeatureMatrix());
        eval.eval(next.getLabels(),output);
        log.info("Score " + eval.stats());
        int[] predict = d.predict(next.getFeatureMatrix());
        log.info("Predict " + Arrays.toString(predict));
    }
}
