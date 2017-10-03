import Jama.Matrix;
import Network.NeuralNet;
import Network.NeuralNetImpl;
import Network.Trainer.NaiveGradientDescent;
import Network.Trainer.Trainer;

import java.util.ArrayList;

import static Ultil.MatrixUtils.normaliseMatrix;

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

        NeuralNet net = new NeuralNetImpl(tIn, tOut, hiddenLayerSizes);
        Trainer trainer = new NaiveGradientDescent(net, tIn, tOut);
        trainer.train();
    }

    public static void main(String args[]){
        NetTester tester = new NetTester();
        tester.basicRun();

    }

}