import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.WorkspaceMode;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.objdetect.Yolo2OutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.modelimport.keras.KerasModelImport;
import org.deeplearning4j.nn.modelimport.keras.exceptions.InvalidKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.exceptions.UnsupportedKerasConfigurationException;
import org.deeplearning4j.nn.transferlearning.FineTuneConfiguration;
import org.deeplearning4j.nn.transferlearning.TransferLearning;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.io.ClassPathResource;
import org.nd4j.linalg.learning.config.Adam;

import java.io.IOException;
import java.util.Random;

public class ImportDarknet {
    public static void main(String[] args) throws java.lang.Exception {

        String filename = new ClassPathResource("tiny_yolo.h5").getFile().getPath();
        ComputationGraph graph = KerasModelImport.importKerasModelAndWeights(filename, false);
        double[][] priorBoxes = {{9.5342,12.8513}, {12.9394,8.8372}, {8.4101,5.1798}, {12.6043,6.2648}, {12.6816,11.2386}};
        INDArray priors = Nd4j.create(priorBoxes);

        int seed = 123;
        Random rng = new Random(seed);

        FineTuneConfiguration fineTuneConf = new FineTuneConfiguration.Builder()
                .seed(seed)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .gradientNormalization(GradientNormalization.RenormalizeL2PerLayer)
                .gradientNormalizationThreshold(1.0)
                .updater(new Adam.Builder().learningRate(1e-3).build())
                //.updater(new Nesterovs.Builder().learningRate(learningRate).momentum(lrMomentum).build())
                .l2(0.00001)
                .activation(Activation.IDENTITY)
                .trainingWorkspaceMode(WorkspaceMode.ENABLED)
                .inferenceWorkspaceMode(WorkspaceMode.ENABLED)
                .build();

        ComputationGraph model = new TransferLearning.GraphBuilder(graph)
                .fineTuneConfiguration(fineTuneConf)
                .addLayer("outputs",
                        new Yolo2OutputLayer.Builder()
                                .boundingBoxPriors(priors)
                                .build(),
                        "conv2d_9")
                .setOutputs("outputs")
                .build();

        System.out.println(model.summary(InputType.convolutional(416, 416, 3)));

        ModelSerializer.writeModel(model, "tiny-yolo-tenun_dl4j_inference.v1.zip", false);
    }

    public ImportDarknet() throws IOException, UnsupportedKerasConfigurationException, InvalidKerasConfigurationException {
    }
}
