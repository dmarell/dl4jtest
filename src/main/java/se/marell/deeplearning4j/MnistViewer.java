/*
 * Created by Daniel Marell 15-02-15 17:24
 */
package se.marell.deeplearning4j;

import javafx.application.Application;
import javafx.event.ActionEvent;
import javafx.event.EventHandler;
import javafx.scene.Scene;
import javafx.scene.control.Button;
import javafx.scene.layout.BorderPane;
import javafx.scene.layout.HBox;
import javafx.stage.Stage;
import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;

import java.io.IOException;

public class MnistViewer extends Application {
    private static int IMAGE_COLS = 20;
    private static int NUM_INPUTS = 1000;

    public static void main(String[] args) {
        launch(args);
    }

    @Override
    public void start(Stage primaryStage) throws IOException {
        DataSetIterator dataSetIter = new MnistDataSetIterator(1, NUM_INPUTS + 1);

        primaryStage.setTitle("MnistViewer");

        HBox buttonBox = new HBox();
        buttonBox.getStyleClass().add("button-pane");

        Button startBtn = createStartButton("Start", actionEvent -> {
            System.out.println("StartBtn");
        });
        buttonBox.getChildren().add(startBtn);

        BorderPane mainPane = new BorderPane();
        mainPane.getStyleClass().add("main-pane");
        mainPane.setCenter(new InputDataSetViewPane(dataSetIter, IMAGE_COLS).build().getNode());
        mainPane.setBottom(buttonBox);

        setEnableState();

        Scene scene = new Scene(mainPane);
        primaryStage.setScene(scene);
        scene.getStylesheets().add(getClass().getResource("/style.css").toExternalForm());
        primaryStage.show();
    }

    private void setEnableState() {
        //TODO
    }

    private Button createStartButton(String caption, EventHandler<ActionEvent> event) {
        Button btn = new Button(caption);
        btn.setOnAction(event);
        return btn;
    }
}
