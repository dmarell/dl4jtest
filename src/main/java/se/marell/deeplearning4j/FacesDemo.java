/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
*/
package se.marell.deeplearning4j;

import org.apache.commons.math3.random.MersenneTwister;
import org.apache.commons.math3.random.RandomGenerator;
import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.LFWDataSetIterator;
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

public class FacesDemo {
    private static Logger log = LoggerFactory.getLogger(FacesDemo.class);

    public static void main(String[] args) throws Exception {
        log.info("#### started");
        RandomGenerator gen = new MersenneTwister(123);

        DataSetIterator fetcher = new LFWDataSetIterator(28, 28);

        NeuralNetConfiguration conf = new NeuralNetConfiguration.Builder()
                .visibleUnit(RBM.VisibleUnit.GAUSSIAN)
                .weightInit(WeightInit.DISTRIBUTION)
                .dist(Distributions.normal(gen, 1e-6))
                .hiddenUnit(RBM.HiddenUnit.RECTIFIED)
                .constrainGradientToUnitNorm(true)
                .activationFunction(Activations.tanh())
                .lossFunction(LossFunctions.LossFunction.RECONSTRUCTION_CROSSENTROPY)
                .rng(gen)
                .learningRate(1e-3f)
                .nIn(fetcher.inputColumns())
                .nOut(fetcher.totalOutcomes())
                .build();

        // Deep Belief Network
        DBN d = new DBN.Builder().configure(conf)
                .hiddenLayerSizes(new int[]{500, 250, 100})
                .build();

        d.getInputLayer().conf().setRenderWeightIterations(5);
        NeuralNetConfiguration.setClassifier(d.getOutputLayer().conf());

        while (fetcher.hasNext()) {
            DataSet dataSet = fetcher.next();
            dataSet.normalizeZeroMeanZeroUnitVariance();
            d.fit(dataSet);

            INDArray predict2 = d.output(dataSet.getFeatureMatrix());
            Evaluation eval = new Evaluation();
            eval.eval(dataSet.getLabels(), predict2);
            log.info(eval.stats());
            int[] predict = d.predict(dataSet.getFeatureMatrix());
            log.info("Predict " + Arrays.toString(predict));
        }
        log.info("#### ready");
    }
}