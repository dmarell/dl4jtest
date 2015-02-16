/*
 * Created by Daniel Marell 15-02-15 17:24
 */
package se.marell.deeplearning4j;

import javafx.application.Application;
import javafx.event.ActionEvent;
import javafx.event.EventHandler;
import javafx.scene.Node;
import javafx.scene.Scene;
import javafx.scene.control.Button;
import javafx.scene.image.ImageView;
import javafx.scene.image.PixelWriter;
import javafx.scene.image.WritableImage;
import javafx.scene.layout.BorderPane;
import javafx.scene.layout.HBox;
import javafx.scene.paint.Color;
import javafx.stage.Stage;
import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;

import java.io.IOException;
import java.util.List;

public class MnistViewer extends Application {
    private static int IMAGE_ROWS = 20;
    private static int IMAGE_COLS = 10;
    private static int IMAGE_SIZE = 28;

    public static void main(String[] args) {
        launch(args);
    }

    private DataSetIterator dataSetIter;

    @Override
    public void start(Stage primaryStage) throws IOException {
        dataSetIter = new MnistDataSetIterator(1, IMAGE_ROWS * IMAGE_COLS + 1);

        primaryStage.setTitle("MnistViewer");

        HBox buttonBox = new HBox();
        buttonBox.getStyleClass().add("button-pane");

        Button nextBtn = createNextButton("Next", actionEvent -> {
        });
        buttonBox.getChildren().add(nextBtn);

        BorderPane mainPane = new BorderPane();
        mainPane.getStyleClass().add("main-pane");
        mainPane.setCenter(createImagePane());
        mainPane.setBottom(buttonBox);


        setEnableState();

        Scene scene = new Scene(mainPane);
        primaryStage.setScene(scene);
        scene.getStylesheets().add(getClass().getResource("/style.css").toExternalForm());
        primaryStage.show();
    }

    private Node createImagePane() {
        final WritableImage image = new WritableImage(IMAGE_SIZE * IMAGE_ROWS, IMAGE_SIZE * IMAGE_COLS);
        PixelWriter pixelWriter = image.getPixelWriter();
        DataSet ds;
        int imageNumber = 0;
        while (dataSetIter.hasNext()) {
            ds = dataSetIter.next();
            System.out.print("numInputs: " + ds.numInputs() + ":");
            INDArray data = ds.getFeatures();

            List<String> labelNames = ds.getLabelNames();
            System.out.print(" [");
            for (String name : labelNames) {
                System.out.print(name + " ");
            }
            System.out.print("] ");

            INDArray labels = ds.getLabels();
            System.out.print(" [");
            for (int i = 0; i < labels.length(); ++i) {
                System.out.print(labels.getInt(i) + " ");
            }
            System.out.print("] ");

            for (int x = 0; x < IMAGE_SIZE; ++x) {
                for (int y = 0; y < IMAGE_SIZE; ++y) {
                    int value = 1 - data.getInt(y * IMAGE_SIZE + x);
                    System.out.print(value + " ");
                    pixelWriter.setColor(IMAGE_SIZE * (imageNumber % IMAGE_ROWS) + x,
                            IMAGE_SIZE * (imageNumber / IMAGE_ROWS) + y,
                            new Color(value, value, value, 1));
                }
            }
            ++imageNumber;
            System.out.println();
        }

        ImageView view = new ImageView(image);
        view.setPreserveRatio(true);
        view.setSmooth(true);
        view.setCache(true);
        HBox box = new HBox();
        box.getStyleClass().add("mnist-image");
        box.getChildren().add(view);
        return box;
    }

    private void setEnableState() {
        //TODO
    }

    private Button createNextButton(String caption, EventHandler<ActionEvent> event) {
        Button btn = new Button(caption);
        btn.setOnAction(event);
        return btn;
    }
}
