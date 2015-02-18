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

import java.io.IOException;

public class NetworkTrainingApp extends Application {
    private static final int IMAGE_COLS = 20;

    private TrainingDataSetViewPane trainingDataSetView;

    public static void main(String[] args) {
        launch(args);
    }

    @Override
    public void start(Stage primaryStage) throws IOException {
        final TrainingSetup trainingSetup = new TrainingSetup(
                dsMap -> Platform.runLater(() -> trainingDataSetView.addResult(dsMap)));

        trainingDataSetView = new TrainingDataSetViewPane(trainingSetup.getInputLabels()).build();

        primaryStage.setTitle("NetworkTrainingApp");

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
        primaryStage.setScene(scene);
        scene.getStylesheets().add(getClass().getResource("/style.css").toExternalForm());
        primaryStage.show();
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
