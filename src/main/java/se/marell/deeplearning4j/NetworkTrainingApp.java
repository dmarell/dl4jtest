/*
 * Created by Daniel Marell 15-02-15 17:24
 */
package se.marell.deeplearning4j;

import javafx.application.Application;
import javafx.application.Platform;
import javafx.event.ActionEvent;
import javafx.event.EventHandler;
import javafx.scene.Node;
import javafx.scene.Scene;
import javafx.scene.control.Button;
import javafx.scene.layout.BorderPane;
import javafx.scene.layout.HBox;
import javafx.stage.Stage;
import javafx.stage.WindowEvent;
import org.nd4j.linalg.dataset.DataSet;

import java.io.IOException;
import java.util.Map;

public class NetworkTrainingApp extends Application {
    private static final int IMAGE_COLS = 20;

    private TrainingDataSetViewPane trainingDataSetView;

    public static void main(String[] args) {
        launch(args);
    }

    @Override
    public void start(Stage stage) throws IOException {
        TrainingSetup trainingSetup = new TrainingSetup((text, dsMap) ->
                Platform.runLater(() -> trainingDataSetView.addResult(text, dsMap)));

        trainingDataSetView = new TrainingDataSetViewPane(trainingSetup.getInputLabels()).build();

        stage.setTitle("NetworkTrainingApp");

        HBox buttonBox = new HBox();
        buttonBox.getStyleClass().add("button-pane");

        Button startBtn = createStartButton("Start", actionEvent -> {
            System.out.println("StartBtn");
            trainingSetup.start();
        });
        buttonBox.getChildren().add(startBtn);

        BorderPane mainPane = new BorderPane();
        mainPane.getStyleClass().add("main-pane");
        mainPane.setCenter(createCenterPane(trainingSetup));
        mainPane.setBottom(buttonBox);

        setEnableState();

        Scene scene = new Scene(mainPane);
        stage.setScene(scene);
        scene.getStylesheets().add(getClass().getResource("/style.css").toExternalForm());
        stage.show();

        stage.setOnCloseRequest(we -> trainingSetup.stop());
    }

    private Node createCenterPane(TrainingSetup trainingSetup) throws IOException {
        Node inputPane = new InputDataSetViewPane(trainingSetup.getInputDataSetIterator(), IMAGE_COLS).build().getNode();

        HBox box = new HBox();
        box.getChildren().add(inputPane);
        box.getChildren().add(trainingDataSetView.getNode());
        return box;
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
