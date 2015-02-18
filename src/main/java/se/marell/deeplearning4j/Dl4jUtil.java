/*
 * Created by Daniel Marell 15-02-17 21:01
 */
package se.marell.deeplearning4j;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;

import java.util.ArrayList;
import java.util.List;

public final class DL4jUtil {
    private DL4jUtil() {
    }

    public static List<String> getLabels(DataSet ds) {
        List<String> result = new ArrayList<>();
        INDArray labels = ds.getLabels();
        for (int i = 0; i < labels.length(); ++i) {
            result.add(labels.getInt(i) + " ");
        }
        return result;
    }

    public static String getLabel(List<String> labelStrings, INDArray labels) {
        for (int i = 0; i < labels.length(); ++i) {
            int v = labels.getInt(i);
            if (v != 0) {
                return labelStrings.get(i);
            }
        }
        return "?";
    }
}
