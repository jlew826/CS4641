//package opt.test;

import dist.*;
import opt.*;
import opt.example.*;
import opt.ga.*;
import shared.*;
import func.nn.backprop.*;

import java.util.*;
import java.io.*;
import java.text.*;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

/**
 * Implementation of randomized hill climbing to
 * find optimal weights to a neural network that is classifying abalone as having either fewer
 * or more than 15 rings.
 *
 * source code adapted from @author Hannah Lau
 * @version 1.0
 */
public class AbaloneTest {
    private static Instance[] instances = initializeInstances();

    private static int inputLayer = 7, hiddenLayer = 5, outputLayer = 1, trainingIterations = 1000;
    private static BackPropagationNetworkFactory factory = new BackPropagationNetworkFactory();

    private static ErrorMeasure measure = new SumOfSquaresError();

    private static DataSet set = new DataSet(instances);

    private static BackPropagationNetwork networks[] = new BackPropagationNetwork[3];
    private static NeuralNetworkOptimizationProblem[] nnop = new NeuralNetworkOptimizationProblem[3];

    private static OptimizationAlgorithm[] oa = new OptimizationAlgorithm[3];
    private static String[] oaNames = {"RHC"};
    private static String results = "";

    private static DecimalFormat df = new DecimalFormat("0.000");

    public static void write_output_to_file(String output_dir, String file_name, String results, boolean final_result) {
        try {
            if (final_result) {
                String augmented_output_dir = output_dir + "/" + new SimpleDateFormat("yyyy-MM-dd").format(new Date());
                String full_path = augmented_output_dir + "/" + file_name;
                Path p = Paths.get(full_path);
                if (Files.notExists(p)) {
                    Files.createDirectories(p.getParent());
                }
                PrintWriter pwtr = new PrintWriter(new BufferedWriter(new FileWriter(full_path, true)));
                synchronized (pwtr) {
                    pwtr.println(results);
                    pwtr.close();
                }
            }
            else {
                String full_path = output_dir + "/" + new SimpleDateFormat("yyyy-MM-dd").format(new Date()) + "/" + file_name;
                Path p = Paths.get(full_path);
                Files.createDirectories(p.getParent());
                Files.write(p, results.getBytes());
            }
        }
        catch (Exception e) {
            e.printStackTrace();
        }

    }



    public static void main(String[] args) {

        String final_result="";

        for(int i = 0; i < oa.length; i++) {
            networks[i] = factory.createClassificationNetwork(
                new int[] {inputLayer, hiddenLayer, outputLayer});
            nnop[i] = new NeuralNetworkOptimizationProblem(set, networks[i], measure);
        }



        int[] iterations = {10, 100, 500, 1000, 2500};

        for (int trainingIterations : iterations) {
          results = "";

          double start = System.nanoTime(), end, trainingTime, testingTime, correct = 0, incorrect = 0;
          oa[0] = new RandomizedHillClimbing(nnop[0]);
          train(oa[0], networks[0], oaNames[0], trainingIterations); //trainer.train();
          end = System.nanoTime();
          trainingTime = end - start;
          trainingTime /= Math.pow(10,9);

          Instance optimalInstance = oa[0].getOptimal();
          networks[0].setWeights(optimalInstance.getData());

          double predicted, actual;
          start = System.nanoTime();
          for(int j = 0; j < instances.length; j++) {
              networks[0].setInputValues(instances[j].getData());
              networks[0].run();

              predicted = Double.parseDouble(instances[j].getLabel().toString());
              actual = Double.parseDouble(networks[0].getOutputValues().toString());

              double trash = Math.abs(predicted - actual) < 0.5 ? correct++ : incorrect++;

            }

            end = System.nanoTime();
            testingTime = end - start;
            testingTime /= Math.pow(10,9);

            results +=  "\nResults for " + oaNames[0] + ": \nCorrectly classified " + correct + " instances." +
                      "\nIncorrectly classified " + incorrect + " instances.\nPercent correctly classified: "
                      + df.format(correct/(correct+incorrect)*100) + "%\nTraining time: " + df.format(trainingTime)
                      + " seconds\nTesting time: " + df.format(testingTime) + " seconds\n";

            final_result = oaNames[0] + "," + "iterations: " + trainingIterations + "," + "training accuracy" + "," + df.format(correct / (correct + incorrect) * 100)
                      + "," + "training time" + "," + df.format(trainingTime) + "," + "testing time" +
                      "," + df.format(testingTime);
            write_output_to_file("Optimization_Results", "abalone_RHC.txt", final_result, true);
        }

        //System.out.println("results for iteration: " + trainingIterations + "---------------------------");
        //System.out.println(results);

    }

    private static void train(OptimizationAlgorithm oa, BackPropagationNetwork network, String oaName, int iteration) {
        //System.out.println("\nError results for " + oaName + "\n---------------------------");
        int trainingIterations = iteration;
        for(int i = 0; i < trainingIterations; i++) {
            oa.train();

            double error = 0;
            for(int j = 0; j < instances.length; j++) {
                network.setInputValues(instances[j].getData());
                network.run();

                Instance output = instances[j].getLabel(), example = new Instance(network.getOutputValues());
                example.setLabel(new Instance(Double.parseDouble(network.getOutputValues().toString())));
                error += measure.value(output, example);
            }

            //System.out.println(df.format(error));
        }
    }

    private static Instance[] initializeInstances() {

        double[][][] attributes = new double[4177][][];

        try {
            BufferedReader br = new BufferedReader(new FileReader(new File("src/opt/jlew7/abalone.txt")));

            for(int i = 0; i < attributes.length; i++) {
                Scanner scan = new Scanner(br.readLine());
                scan.useDelimiter(",");

                attributes[i] = new double[2][];
                attributes[i][0] = new double[7]; // 7 attributes
                attributes[i][1] = new double[1];

                for(int j = 0; j < 7; j++)
                    attributes[i][0][j] = Double.parseDouble(scan.next());

                attributes[i][1][0] = Double.parseDouble(scan.next());
            }
        }
        catch(Exception e) {
            e.printStackTrace();
        }

        Instance[] instances = new Instance[attributes.length];

        for(int i = 0; i < instances.length; i++) {
            instances[i] = new Instance(attributes[i][0]);
            // classifications range from 0 to 30; split into 0 - 14 and 15 - 30
            instances[i].setLabel(new Instance(attributes[i][1][0] < 15 ? 0 : 1));
        }

        return instances;
    }
}
