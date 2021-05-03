import org.bytedeco.opencv.opencv_core.Scalar;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.transform.ColorConversionTransform;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.layers.objdetect.DetectedObject;
import org.deeplearning4j.nn.layers.objdetect.YoloUtils;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.factory.Nd4j;

import java.io.File;
import java.io.FileWriter;
import java.io.FilenameFilter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.Scanner;

import static org.opencv.imgproc.Imgproc.COLOR_BGR2RGB;

public class EvaluateTestData {
    // File representing the folder that you select using a FileChooser
    static final File dir = new File("Dataset\\Test\\JPEGImages");

    // array of supported extensions (use a List if you prefer)
    static final String[] EXTENSIONS = new String[]{
            "gif", "png", "bmp", "jpg" // and other formats you need
    };
    // filter to identify images based on their extensions
    static final FilenameFilter IMAGE_FILTER = new FilenameFilter() {

        @Override
        public boolean accept(final File dir, final String name) {
            for (final String ext : EXTENSIONS) {
                if (name.endsWith("." + ext)) {
                    return (true);
                }
            }
            return (false);
        }
    };

    public static void main(String[] args) throws IOException {
        //Atur parameter klasifikasi
        ComputationGraph model;
        double detectionThreshold = 0.5;
        double nmsThreshold = 0.25;
        String modelPath = "Model\\model.zip";
        String clasessPath = "Model\\classes.txt";
        double[][] priorBoxes = {{9.5342,12.8513}, {12.9394,8.8372}, {8.4101,5.1798}, {12.6043,6.2648}, {12.6816,11.2386}};
        INDArray priors = Nd4j.create(priorBoxes);
        double width = 416;
        double height = 416;

        if (dir.isDirectory()) { // make sure it's a directory
            FileWriter myWriter = new FileWriter("result.txt");
            int evalCount = 0;
            int detectCount = 0;
            int Maincount = 0;
            double Mainavr = 0, Mainhigh = 0, Mainlow= 1, FalseCount = 0;

            for (final File gambar : Objects.requireNonNull(dir.listFiles(IMAGE_FILTER))) {
                boolean detected = false;
                evalCount += 1;

                //Load Gambar
                NativeImageLoader imageLoader = new NativeImageLoader((long) height, (long) width, 3, new ColorConversionTransform(COLOR_BGR2RGB));
                INDArray image = imageLoader.asMatrix(gambar);

                //Normalisasikan Nilai Gambar
                DataNormalization scalar = new ImagePreProcessingScaler(0, 1);
                scalar.transform(image);

                //Load Model
                File file = new File(modelPath);
                model = ComputationGraph.load(file, false);

                //Load Label
                File labelFile = new File(clasessPath);
                Scanner s = new Scanner(labelFile);
                ArrayList<String> labels = new ArrayList<String>();
                while (s.hasNextLine()){
                    labels.add(s.nextLine());
                }
                s.close();

                //Klasifikasikan
                INDArray results = model.outputSingle(image);
                List<DetectedObject> objs = YoloUtils.getPredictedObjects(priors,results,detectionThreshold,nmsThreshold);

                for (DetectedObject obj : objs) {
                    //Dapatkan Tingkat Keyakikan dan Label objek yang terdeteksi
                    String confident = String.valueOf((int) (obj.getConfidence() * 100));
                    String label = labels.get(obj.getPredictedClass());

                    System.out.println(gambar.getName()+ " : " +label +" - "+confident+"%");
                    myWriter.write(gambar.getName()+ " : " +label +" - "+confident+"%\n");

                    String[] arrLabel = label.split(" ", 2);
                    String[] arrName = gambar.getName().split("_", 2);
                    if(arrLabel[0].equals(arrName[0])){
                        detected = true;
                        Maincount += 1;

                        Mainavr += obj.getConfidence();
                        if(obj.getConfidence() < Mainlow){Mainlow = obj.getConfidence();}
                        if(obj.getConfidence() > Mainhigh){Mainhigh = obj.getConfidence();}
                    }else {
                        FalseCount += 1;
                    }
                }
                if(detected){detectCount+=1;}
            }
            Mainavr = Mainavr/Maincount;
            Mainavr = (double) Math.round(Mainavr * 10000) / 100;
            Mainlow = (double) Math.round(Mainlow * 10000) / 100;
            Mainhigh = (double) Math.round(Mainhigh * 10000) / 100;

            System.out.println("Test Data Confident :");
            System.out.println("Average : " + (Mainavr) + "%    Lowest : " + (Mainlow) + "%  Highest : " + (Mainhigh)+ "%");
            myWriter.write("Average : " + (Mainavr) + "%    Lowest : " + (Mainlow) + "%  Highest : " + (Mainhigh)+ "%\n");

            System.out.println("Test Data Evaluation :");
            System.out.println("Total Test Data : " + evalCount + "    Total Detected Test Data : " + detectCount + "  Model Accuracy : " + (double) detectCount/evalCount);
            myWriter.write("Total Test Data : " + evalCount + "    Total Detected Test Data : " + detectCount + "  Model Accuracy : " + (double) detectCount/evalCount + "\n");

            System.out.println("Total Object Detected :" + Maincount);
            System.out.println("Total False Detection :" + (Maincount-detectCount));

            myWriter.write("Total Object Detected :" + Maincount + "\n");
            myWriter.write("Total False Detection :" + FalseCount + "\n");
            myWriter.close();
        }
    }
}
