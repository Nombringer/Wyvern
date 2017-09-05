import Jama.Matrix;

import java.util.ArrayList;
import java.util.Arrays;

/**
 * Created by fabd on 10/08/17.
 */


public class NeuralNet {
    private boolean testMode = true;
    private static final double[][] trainingIn =  {{3, 5}, {5, 1}, {10, 2}};
    private static final double[]   trainingOut = {75, 82, 93};


    private int inputLayerSize;
    private int outputLayerSize;
    private int hiddenLayerSize;
    private ActivationFunction activationFunction = new Sygmoid();

    private ArrayList<Matrix> weights = new ArrayList<>();
    private Matrix inputData;
    private Matrix targetData;

     NeuralNet(){
         if(!testMode){
             setHyperParameters();
         }else{

             inputLayerSize = 2;
             outputLayerSize = 1;
             hiddenLayerSize = 3;
         }
         generateWeights();
         generateTrainingMatrices();

         Matrix yhat = forward();
         Matrix error = getSqrError(yhat);
         System.out.println("SquaredError :");
         error.print(1, 3);
         System.out.println("Target Data: ");
         targetData.print(1, 3);
    }

    private Matrix forward(){

        System.out.println("Beginning forward propagation... \n");

        System.out.println("Training Data:");
        inputData.print(1, 5);

        System.out.println("Weight matrices:");

        for(Matrix m : weights){
            m.print(1, 5);
        }

        Matrix z2 = new Matrix(inputData.times(weights.get(0)).getArray());
        System.out.println("z2");
        z2.print(1, 5);

        Matrix a2 = activationFunction.apply(z2);
        System.out.println("a2");
        a2.print(1, 5);

        Matrix z3 = new Matrix(a2.times(weights.get(1)).getArray());
        System.out.println("z3");
        z3.print(1, 5);

        Matrix yhat = activationFunction.apply(z3);
        System.out.println("Forward Propagation results :");
        yhat.print(1, 5);

        return yhat;
    }

    private Matrix backPropagate(Matrix estimated){







    }

    private void generateTrainingMatrices(){
        inputData = normaliseMatrix(new Matrix(trainingIn));
        targetData = normaliseMatrix(new Matrix(trainingOut, 3));
        System.out.println("Normalised Data: ");
        inputData.print(1, 3);
        targetData.print(1, 3);

    }

    private Matrix getSqrError(Matrix estimates){
        Matrix error = targetData.minus(estimates);
        return error.arrayTimesEquals(targetData.minus(estimates));
    }

    /**
     * Generate the weights based on the the HyperParameters
     */
    private void generateWeights(){
       //TODO: Make this actually use the hyperparams

       Matrix W1 =  Matrix.random(inputLayerSize, hiddenLayerSize);
       Matrix W2 =  Matrix.random(hiddenLayerSize, outputLayerSize);

       System.out.println("Randomly generated weights:");

       W1.print(1, 3);
       W2.print(1, 3);

       weights.add(W1); weights.add(W2);

    }


    /**
     * Set the HyperParameters based on console input
     */
    private void setHyperParameters(){

        System.out.println("Setting HyperParameters: \n Please enter the input layer size:");
        inputLayerSize = Integer.parseInt(System.console().readLine());

        System.out.println("Please enter the outputLayer size");
        outputLayerSize = Integer.parseInt(System.console().readLine());

        System.out.println("Please enter the Hidden Layer size");
        hiddenLayerSize = Integer.parseInt(System.console().readLine());
    }
    ///////////////////////////
    //////HELPER FUNCTIONS/////
    ///////////////////////////


    public static double getMax(Matrix matrix) {
        double max = Double.MIN_VALUE;

        for (int i = 0; i < matrix.getRowDimension(); i++) {
            for (int j = 0; j < matrix.getColumnDimension(); j++) {

                max = Math.max(max, matrix.get(i, j));
            }

        }
        return max;
    }
    /**
     *
     * @param matrix
     * @return The matrix, with the data normalised to 1
     */
    public static Matrix normaliseMatrix(Matrix matrix){
        System.out.println("Normalising, max = " + getMax(matrix));
        Matrix normalised = matrix.times(1/getMax(matrix));
        return normalised;
    }

    ///////////////////////////
    /////////Tests/////////////
    ///////////////////////////


    private void testSygmoid(){
        System.out.println("Testing sygmoid: ");
        System.out.println("Sygmoid(1) = " + activationFunction.apply(1) );
        System.out.println("Sygmoid(-1, 0 1) = ");
        Matrix out = activationFunction.apply(new Matrix( new double[] {-1, 0, 1}, 1 ));
        out.print(1, 2);

        System.out.println("Testing sygmoidPrime: ");
        System.out.println("Sygmoidprime(1) = " + activationFunction.applyGradFunc(1) );
        System.out.println("Sygmoid(-1, 0 1) = ");
        Matrix out2 = activationFunction.applyGradFunc(new Matrix( new double[] {-1, 0, 1}, 1 ));
        out2.print(1, 2);


    }



    public static void main(String args[]) {
        NeuralNet wyvren = new NeuralNet();
    }


}
