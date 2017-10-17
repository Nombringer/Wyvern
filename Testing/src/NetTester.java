import Jama.Matrix;
import Network.NeuralNet;
import Network.MultilayerPerceptron;
import Network.Trainer.NaiveGradientDescent;
import Network.Trainer.SGDTrainer;
import Network.Trainer.Trainer;

import java.util.ArrayList;
import java.util.Random;

import static Util.MatrixUtils.normaliseMatrix;

/**
 * Created by fabd on 3/10/17.
 */
public class NetTester {

    private final double[][] trainingIn =  {{3, 5}, {5, 1}, {10, 2}};
    private final double[]   trainingOut = {75, 82, 93};


    void basicRun(){

        Matrix tIn = normaliseMatrix(new Matrix(trainingIn));
        Matrix tOut = normaliseMatrix(new Matrix(trainingOut, 3));
        ArrayList<Integer> hiddenLayerSizes = new ArrayList<>();
        hiddenLayerSizes.add(10);

        NeuralNet net = new MultilayerPerceptron(tIn, tOut, hiddenLayerSizes);
        Trainer trainer = new NaiveGradientDescent(net, tIn, tOut);
        trainer.train();
    }

    public static void main(String args[]){
        NetTester tester = new NetTester();
        tester.learnNothing();

    }

    void learnNothing(){
        int examplenum = 10000;

        Random random = new Random();
        double[][] input = new double[examplenum][2];
        double[] output = new double[examplenum];

        for (int i = 0; i < examplenum; i++) {
            for (int j = 0; j < 1 ; j++) {
                input[i][j] = random.nextDouble()*5;
            }
            output[i] = random.nextDouble();
        }

        Matrix tIn = normaliseMatrix(new Matrix(input));
        Matrix tOut = normaliseMatrix(new Matrix(output, examplenum));

        ArrayList layers = new ArrayList(); layers.add(3);

        NeuralNet net = new MultilayerPerceptron(tIn, tOut, layers);
        SGDTrainer trainer = new SGDTrainer(net, tIn, tOut, 20);
        trainer.train();

    }

}
