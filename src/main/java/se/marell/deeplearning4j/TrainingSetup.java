/*
 * Created by Daniel Marell 15-02-17 21:51
 */
package se.marell.deeplearning4j;

import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.nd4j.linalg.dataset.DataSet;

import java.io.IOException;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class TrainingSetup {
    private static final int NUM_INPUTS = 1000;

    public interface Listener {
        void dataSetUpdated(Map<String, DataSet> dsMap);
    }

    private Listener listener;
    private List<String> inputLabels = Arrays.asList("0", "1", "2", "3", "4", "5", "6", "7", "8", "9");
    private DataSetIterator inputDataSetIterator;
//    private DataSetIterator trainingDataSetIterator;

    public TrainingSetup(Listener listener) throws IOException {
        this.listener = listener;
        inputDataSetIterator = new MnistDataSetIterator(1, NUM_INPUTS + 1);
    }

    public DataSetIterator getInputDataSetIterator() throws IOException {
        return inputDataSetIterator;
    }

    public List<String> getInputLabels() {
        return inputLabels;
    }

    public void start() {
        final Map<String, DataSet> dsMap = createDsMap();
        Thread t = new Thread(() -> {
            while (true) {
                Map<String, DataSet> morphedMsMap = morphDsMap(dsMap);
                listener.dataSetUpdated(morphedMsMap);
                try {
                    Thread.sleep(1000);
                } catch (InterruptedException ignore) {
                }
            }
        });
        t.start();
    }

    private Map<String, DataSet> createDsMap() {
        Map<String, DataSet> result = new HashMap<>();
        for (String label : inputLabels) {
            result.put(label, null);
        }
        inputDataSetIterator.reset();
        while (inputDataSetIterator.hasNext()) {
            DataSet ds = inputDataSetIterator.next();
            String s = DL4jUtil.getLabel(inputLabels, ds.getLabels());
            result.put(s, ds);
            if (allBucketsInitialized(result)) {
                break;
            }
        }
        inputDataSetIterator.reset();
        return result;
    }

    private boolean allBucketsInitialized(Map<String, DataSet> result) {
        for (DataSet ds : result.values()) {
            if (ds == null) {
                return false;
            }
        }
        return true;
    }

    private Map<String, DataSet> morphDsMap(Map<String, DataSet> dsMap) {
        return dsMap;
    }
}
