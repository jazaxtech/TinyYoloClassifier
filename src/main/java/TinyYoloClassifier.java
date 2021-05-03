import org.bytedeco.javacv.CanvasFrame;
import org.bytedeco.javacv.OpenCVFrameConverter;
import org.bytedeco.opencv.global.opencv_core;
import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_core.Point;
import org.bytedeco.opencv.opencv_core.Scalar;
import org.bytedeco.opencv.opencv_core.Size;
import org.datavec.api.records.metadata.RecordMetaDataImageURI;
import org.datavec.api.split.FileSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.objdetect.ObjectDetectionRecordReader;
import org.datavec.image.recordreader.objdetect.impl.VocLabelProvider;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.WorkspaceMode;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.objdetect.Yolo2OutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.layers.objdetect.DetectedObject;
import org.deeplearning4j.nn.transferlearning.FineTuneConfiguration;
import org.deeplearning4j.nn.transferlearning.TransferLearning;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.EvaluativeListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.FileStatsStorage;
import org.deeplearning4j.util.ModelSerializer;
import org.deeplearning4j.zoo.model.TinyYOLO;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.List;
import java.util.Random;
import java.util.concurrent.TimeUnit;

import static org.bytedeco.opencv.global.opencv_core.CV_8U;
import static org.bytedeco.opencv.global.opencv_core.RGB;
import static org.bytedeco.opencv.global.opencv_imgproc.*;

public class TinyYoloClassifier{
    private static final Logger log = LoggerFactory.getLogger(TinyYoloClassifier.class);

    // Enable different colour bounding box for different classes
    public static final Scalar RED = RGB(255.0, 0, 0);
    public static final Scalar GREEN = RGB(0, 255.0, 0);
    public static final Scalar BLUE = RGB(0, 0, 255.0);
    public static final Scalar YELLOW = RGB(255.0, 255.0, 0);
    public static final Scalar CYAN = RGB(0, 255.0, 255.0);
    public static final Scalar MAGENTA = RGB(255.0, 0.0, 255.0);
    public static final Scalar ORANGE = RGB(255.0, 128.0, 0);
    public static final Scalar PINK = RGB(255.0, 192.0, 203.0);
    public static final Scalar LIGHTBLUE = RGB(153.0, 204.0, 255.0);
    public static final Scalar VIOLET = RGB(238.0, 130.0, 238.0);
    public static final Scalar WHITE = RGB(255,255,255);

    private static String formatInterval(final long l)
    {
        final long hours = TimeUnit.SECONDS.toHours(l);
        final long mins = TimeUnit.SECONDS.toMinutes(l - TimeUnit.HOURS.toSeconds(hours));
        final long secs = TimeUnit.SECONDS.toSeconds(l - TimeUnit.HOURS.toSeconds(hours) - TimeUnit.MINUTES.toSeconds(mins));
        return String.format("%02d hrs %02d mins %02d secs", hours, mins, secs);
    }

    public static void main(String[] args) throws java.lang.Exception {

        // parameters for the training phase
        int batchSize = 10;
        int nEpochs = 1 ;
        double learningRate = 1e-4;
        double detectionThreshold = 0.3;
        String trainingTypes = "new";
        String trainingDataset = "same";
        int backup = 0;
        boolean visualize = true;

        // number of classes for the datasets
        int nClasses = 1;

        // arguments variable
        if(args.length == 0)
        {
            System.out.println("Improper use of the program, please read the tutorial on how to use this program correctly");
            System.exit(0);
        }
        else {
            nEpochs = Integer.parseInt(args[0]);
            batchSize = Integer.parseInt(args[1]);
            nClasses = Integer.parseInt(args[2]);
            learningRate = Double.parseDouble(args[3]);
            detectionThreshold = Double.parseDouble(args[4]);
            trainingTypes = args[5];
            trainingDataset = args[6];
            backup = Integer.parseInt(args[7]);
            visualize = Boolean.parseBoolean(args[8]);
        }

        // parameters matching the pretrained TinyYOLO model
        int width = 416;
        int height = 416;
        int nChannels = 3;
        int gridWidth = 13;
        int gridHeight = 13;

        // parameters for the Yolo2OutputLayer
        int nBoxes = 5;
        double lambdaNoObj = 0.5;
        double lambdaCoord = 5.0;
        double[][] priorBoxes = {{2, 5}, {2.5, 6}, {3, 7}, {3.5, 8}, {4, 9}};

        int seed = 123;
        Random rng = new Random(seed);

        String dataTrainDir = "Dataset\\Train";
        File trainDir = new File(dataTrainDir + "\\JPEGImages");
        FileSplit trainData = new FileSplit(trainDir, NativeImageLoader.ALLOWED_FORMATS, rng);

        String dataTestDir = "Dataset\\Test";
        File testDir = new File(dataTestDir + "\\JPEGImages");
        FileSplit testData = new FileSplit(testDir, NativeImageLoader.ALLOWED_FORMATS, rng);

        if(trainingDataset.equals("same")){
            dataTestDir = dataTrainDir;
            testData = trainData;
        }

        //Initialize the user interface backend
        UIServer uiServer = UIServer.getInstance();
        //Configure where the network information (gradients, score vs. time etc) is to be stored.
        StatsStorage statsStorage = new FileStatsStorage(new File("TrainStats\\TrainingStats-UIServer.dl4j"));
        //Attach the StatsStorage instance to the UI: this allows the contents of the StatsStorage to be visualized
        uiServer.attach(statsStorage);

        // Setup RecordReader for the Annotations
        ObjectDetectionRecordReader recordReaderTrain = new ObjectDetectionRecordReader(height, width, nChannels,
                gridHeight, gridWidth, new VocLabelProvider(dataTrainDir));
        recordReaderTrain.initialize(trainData);
        ObjectDetectionRecordReader recordReaderTest = new ObjectDetectionRecordReader(height, width, nChannels,
                gridHeight, gridWidth, new VocLabelProvider(dataTestDir));
        recordReaderTest.initialize(testData);

        // ObjectDetectionRecordReader performs regression, so we need to specify it here
        RecordReaderDataSetIterator train = new RecordReaderDataSetIterator(recordReaderTrain, batchSize, 1, 1, true);
        train.setPreProcessor(new ImagePreProcessingScaler(0, 1));

        RecordReaderDataSetIterator test = new RecordReaderDataSetIterator(recordReaderTest, 1, 1, 1, true);
        test.setPreProcessor(new ImagePreProcessingScaler(0, 1));

        ComputationGraph model;
        String modelFilename = "Model\\model.zip";

        long startTime = System.nanoTime();

        if(trainingTypes.equals("new")){
            log.info("Build model...");

            ComputationGraph pretrained = (ComputationGraph) TinyYOLO.builder().build().initPretrained();
            INDArray priors = Nd4j.create(priorBoxes);

            FineTuneConfiguration fineTuneConf = new FineTuneConfiguration.Builder()
                    .seed(seed)
                    .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                    .gradientNormalization(GradientNormalization.RenormalizeL2PerLayer)
                    .gradientNormalizationThreshold(1.0)
                    .updater(new Adam.Builder().learningRate(learningRate).build())
                    //.updater(new Nesterovs.Builder().learningRate(learningRate).momentum(lrMomentum).build())
                    .l2(0.00001)
                    .activation(Activation.IDENTITY)
                    .trainingWorkspaceMode(WorkspaceMode.ENABLED)
                    .inferenceWorkspaceMode(WorkspaceMode.ENABLED)
                    .build();

            model = new TransferLearning.GraphBuilder(pretrained)
                    .fineTuneConfiguration(fineTuneConf)
                    .removeVertexKeepConnections("conv2d_9")
                    .removeVertexKeepConnections("outputs")
                    .addLayer("convolution2d_9",
                            new ConvolutionLayer.Builder(1,1)
                                    .nIn(1024)
                                    .nOut(nBoxes * (5 + nClasses))
                                    .stride(1,1)
                                    .convolutionMode(ConvolutionMode.Same)
                                    .weightInit(WeightInit.XAVIER)
                                    .activation(Activation.IDENTITY)
                                    .build(),
                            "leaky_re_lu_8")
                    .addLayer("outputs",
                            new Yolo2OutputLayer.Builder()
                                    .lambdaNoObj(lambdaNoObj)
                                    .lambdaCoord(lambdaCoord)
                                    .boundingBoxPriors(priors)
                                    .build(),
                            "convolution2d_9")
                    .setOutputs("outputs")
                    .build();
        } else {
            log.info("Load model...");
            model = ComputationGraph.load(new File(modelFilename), true);
        }

        System.out.println(model.summary(InputType.convolutional(height, width, nChannels)));
        model.setListeners(new ScoreIterationListener(1),new StatsListener(statsStorage));

        if(nEpochs > 0){
            log.info("Train model...");
            for (int i = 0; i < nEpochs; i++) {
                while (train.hasNext()) {
                    DataSet d = train.next();
                    model.fit(d);
                }
                train.reset();
                log.info("End of epoch # " + (i+1));
                if(backup >= 1){
                    if((i+1) % backup == 0){
                        log.info("Save backup model...");
                        ModelSerializer.writeModel(model, "Model\\Backup\\backupModel-"+(i+1)+".zip" , true);

                        try(BufferedWriter writer = new BufferedWriter(new FileWriter("Model\\classes.txt"))) {
                            log.info("Save classes names...");
                            List<String> labels = train.getLabels();
                            for (String label : labels) {
                                writer.write(label);
                                writer.write("\n");
                            }

                        }
                        catch(IOException e){
                            log.info("Can't save classes names...");
                        }
                    }
                }
            }

            log.info("Save model...");
            ModelSerializer.writeModel(model, modelFilename, true);

            try(BufferedWriter writer = new BufferedWriter(new FileWriter("Model\\classes.txt"))) {
                log.info("Save classes names...");
                List<String> labels = train.getLabels();
                for (String label : labels) {
                    writer.write(label);
                    writer.write("\n");
                }

            }
            catch(IOException e){
                log.info("Can't save classes names...");
            }

        }

        long endTime = System.nanoTime();
        long timeElapsed = endTime - startTime;
        timeElapsed = TimeUnit.NANOSECONDS.toSeconds(timeElapsed);
        String time = formatInterval(timeElapsed);
        log.info("Elapsed time = " + time);

        // visualize results on the test set
        NativeImageLoader imageLoader = new NativeImageLoader();
        CanvasFrame frame = new CanvasFrame("TinyYoloClassifier");
        OpenCVFrameConverter.ToMat converter = new OpenCVFrameConverter.ToMat();
        org.deeplearning4j.nn.layers.objdetect.Yolo2OutputLayer yout =
                (org.deeplearning4j.nn.layers.objdetect.Yolo2OutputLayer)model.getOutputLayer(0);
        List<String> labels = train.getLabels();
        test.setCollectMetaData(true);
        Scalar[] colormap = {RED,BLUE,GREEN,CYAN,YELLOW,MAGENTA,ORANGE,PINK,LIGHTBLUE,VIOLET};

        int evalCount = 0;
        int detectCount = 0;
        int Maincount = 0;
        double Mainavr = 0, Mainhigh = 0, Mainlow= 1;
        while (test.hasNext() && frame.isVisible()) {
            evalCount += 1;
            org.nd4j.linalg.dataset.DataSet ds = test.next();
            RecordMetaDataImageURI metadata = (RecordMetaDataImageURI)ds.getExampleMetaData().get(0);
            INDArray features = ds.getFeatures();
            INDArray results = model.outputSingle(features);
            List<DetectedObject> objs = yout.getPredictedObjects(results, detectionThreshold);
            File file = new File(metadata.getURI());
            log.info(file.getName() + ": " + objs);

            Mat mat = imageLoader.asMat(features);
            Mat convertedMat = new Mat();
            mat.convertTo(convertedMat, CV_8U, 255, 0);
            int w = metadata.getOrigW() * 2;
            int h = metadata.getOrigH() * 2;
            Mat image = new Mat();
            resize(convertedMat, image, new Size(w, h));

            Mat finalImage = new Mat();
            int padding = 0;
            opencv_core.copyMakeBorder( image, finalImage, padding, padding, padding, padding, opencv_core.BORDER_CONSTANT, WHITE);

            int count = 0;
            double avr = 0, high = 0, low= 1;
            boolean detected = false;
            for (DetectedObject obj : objs) {
                detected = true;
                Maincount += 1;
                count += 1;
                double[] xy1 = obj.getTopLeftXY();
                double[] xy2 = obj.getBottomRightXY();
                String confident = String.valueOf((int) (obj.getConfidence() * 100));
                String label = labels.get(obj.getPredictedClass());
                int x1 = (int) Math.round(w * xy1[0] / gridWidth);
                int y1 = (int) Math.round(h * xy1[1] / gridHeight);
                int x2 = (int) Math.round(w * xy2[0] / gridWidth);
                int y2 = (int) Math.round(h * xy2[1] / gridHeight);
                int color = obj.getPredictedClass();
                if (color >= 20) {
                    color -= 20;
                }
                else if (color >= 10) {
                    color -= 10;
                }
                rectangle(finalImage, new Point(x1, y1), new Point(x2, y2), colormap[color]);
                putText(finalImage, label + "("+ obj.getPredictedClass() + ")", new Point((x1 + x2)/2, (y1+y2)/2), FONT_HERSHEY_DUPLEX, 1, colormap[color]);
                putText(finalImage, confident + " %" + "(" + obj.getClassPredictions()+ ")", new Point((x1 + x2)/2, (y1+y2)/2 + 30), FONT_HERSHEY_DUPLEX, 1, colormap[color]);

                avr += obj.getConfidence();
                if(obj.getConfidence() < low){low = obj.getConfidence();}
                if(obj.getConfidence() > high){high = obj.getConfidence();}

                Mainavr += obj.getConfidence();
                if(obj.getConfidence() < Mainlow){Mainlow = obj.getConfidence();}
                if(obj.getConfidence() > Mainhigh){Mainhigh = obj.getConfidence();}
            }
            if(detected){detectCount+=1;}
            avr = avr/count;
            avr = (double) Math.round(avr * 10000) / 100;
            low = (double) Math.round(low * 10000) / 100;
            high = (double) Math.round(high * 10000) / 100;
            System.out.println("Confident :");
            System.out.println("Average : " + (avr) + "%    Lowest : " + (low) + "%  Highest : " + (high) +"%");

            if(visualize) {
                frame.setTitle(new File(metadata.getURI()).getName() + " - TinyYoloClassifier");
                frame.setCanvasSize(w, h);
                frame.showImage(converter.convert(finalImage));
                frame.waitKey();
                System.out.println("Width :" + w + "Height" + h);
            }
        }
        Mainavr = Mainavr/Maincount;
        Mainavr = (double) Math.round(Mainavr * 10000) / 100;
        Mainlow = (double) Math.round(Mainlow * 10000) / 100;
        Mainhigh = (double) Math.round(Mainhigh * 10000) / 100;

        System.out.println("Test Data Confident :");
        System.out.println("Average : " + (Mainavr) + "%    Lowest : " + (Mainlow) + "%  Highest : " + (Mainhigh)+ "%");

        System.out.println("Test Data Evaluation :");
        System.out.println("Total Test Data : " + evalCount + "    Total Detected Test Data : " + detectCount + "  Model Accuracy : " + (double) detectCount/evalCount);

        frame.dispose();
    }
}