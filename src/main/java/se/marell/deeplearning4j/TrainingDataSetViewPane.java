/*
 * Created by Daniel Marell 15-02-16 22:17
 */
package se.marell.deeplearning4j;

import javafx.geometry.Insets;
import javafx.scene.Node;
import javafx.scene.control.ScrollPane;
import javafx.scene.image.ImageView;
import javafx.scene.image.PixelWriter;
import javafx.scene.image.WritableImage;
import javafx.scene.layout.GridPane;
import javafx.scene.layout.VBox;
import javafx.scene.paint.Color;
import javafx.scene.text.Text;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;

import java.util.List;
import java.util.Map;

public class TrainingDataSetViewPane {
    private GridPane grid;
    private ScrollPane scrollPane;
    private List<String> labels;
    private int rowCount;

    public TrainingDataSetViewPane(List<String> labels) {
        this.labels = labels;

        grid = new GridPane();
        grid.setHgap(3);
        grid.setVgap(3);
        grid.setPadding(new Insets(0, 3, 0, 3));

        scrollPane = new ScrollPane();
        scrollPane.setContent(grid);
        scrollPane.getStyleClass().add("train-scroll-pane");
    }

    public void addResult(Map<String, DataSet> dsMap) {
        int col = 0;
        for (String label : dsMap.keySet()) {
            grid.add(createLabel(label), ++col, 0);
        }

        col = 0;
        for (String label : labels) {
            DataSet ds = dsMap.get(label);
            if (ds != null) {
                grid.add(createCell(ds), col, rowCount + 1);
            }
            ++col;
        }
        ++rowCount;
    }

    private Node createLabel(String label) {
        Text text = new Text(label);
        text.getStyleClass().add("train-label");
        return text;
    }

    private Node createCell(DataSet ds) {
        int size = (int) Math.sqrt(ds.numInputs());
        final WritableImage image = new WritableImage(size, size);
        PixelWriter pixelWriter = image.getPixelWriter();

        INDArray data = ds.getFeatures();

        ImageView view = new ImageView(image);
        VBox box = new VBox();
        box.getStyleClass().add("training-image");
        box.getChildren().add(view);
        box.setPadding(new Insets(3, 3, 0, 0));

        for (int x = 0; x < size; ++x) {
            for (int y = 0; y < size; ++y) {
                int value = 1 - data.getInt(y * size + x);
                pixelWriter.setColor(x, y, new Color(value, value, value, 1));
            }
        }
        return box;
    }

    public TrainingDataSetViewPane build() {
        return this;
    }

    public Node getNode() {
        return scrollPane;
    }
}
