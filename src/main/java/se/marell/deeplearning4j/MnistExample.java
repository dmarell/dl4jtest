package se.marell.deeplearning4j;

import org.apache.commons.math3.random.MersenneTwister;
import org.apache.commons.math3.random.RandomGenerator;
import org.deeplearning4j.datasets.fetchers.MnistDataFetcher;
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
public class MnistExample {

    private static Logger log = LoggerFactory.getLogger(MnistExample.class);

    public static void main(String[] args) throws Exception {
        RandomGenerator gen = new MersenneTwister(123);

//        NeuralNetConfiguration conf = new NeuralNetConfiguration.Builder()
//                .visibleUnit(RBM.VisibleUnit.GAUSSIAN)
//                .weightInit(WeightInit.DISTRIBUTION)
//                .dist(Distributions.normal(gen, 1e-6))
//                .hiddenUnit(RBM.HiddenUnit.RECTIFIED)
//                .constrainGradientToUnitNorm(true)
//                .activationFunction(Activations.tanh())
//                .lossFunction(LossFunctions.LossFunction.RECONSTRUCTION_CROSSENTROPY)
        NeuralNetConfiguration conf = new NeuralNetConfiguration.Builder()
//                .momentum(5e-1f)
                .constrainGradientToUnitNorm(false)
                .weightInit(WeightInit.DISTRIBUTION)
//                .dist(Distributions.normal(gen, 1e-6))
//                .hiddenUnit(RBM.HiddenUnit.RECTIFIED)
                .iterations(20)
                .withActivationType(NeuralNetConfiguration.ActivationType.SAMPLE)
                .lossFunction(LossFunctions.LossFunction.RECONSTRUCTION_CROSSENTROPY)
                .rng(gen)
                .learningRate(1e-2f)
                .nIn(784)
                .nOut(10)
                .build();

        DBN d = new DBN.Builder().configure(conf)
                .hiddenLayerSizes(new int[]{500, 250, 100})
                .build();

        d.getInputLayer().conf().setRenderWeightIterations(10);
        d.getOutputLayer().conf().setNumIterations(10);
        NeuralNetConfiguration.setClassifier(d.getOutputLayer().conf());

        MnistDataFetcher fetcher = new MnistDataFetcher(true);
        fetcher.fetch(1000);
        DataSet d2 = fetcher.next();
        d2.normalizeZeroMeanZeroUnitVariance();

        d.fit(d2);

        INDArray predict2 = d.output(d2.getFeatureMatrix());

        Evaluation eval = new Evaluation();
        eval.eval(d2.getLabels(), predict2);
        log.info(eval.stats());
        int[] predict = d.predict(d2.getFeatureMatrix());
        log.info("Predict " + Arrays.toString(predict));
    }
}
