import Jama.Matrix;

import java.util.ArrayList;
import java.util.Random;

/**
 * Base Neural Net class for project Wyvern.
 */

//TODO: Implement early stopping and regularisation
@SuppressWarnings({"FieldCanBeLocal", "WeakerAccess", "JavaDoc"})
public class NeuralNet {

    private static final double[][] trainingIn =  {{3, 5}, {5, 1}, {10, 2}};
    private static final double[]   trainingOut = {75, 82, 93};

    private int hiddenLayerNum = 1;
    private int inputLayerSize;
    private int outputLayerSize;
    private int hiddenLayerSize;
    private ActivationFunction activationFunction = new HyperbolicTangent();

    private ArrayList<Matrix> weights = new ArrayList<>();
    private ArrayList<Matrix> weightCostGradient = new ArrayList<>(2);
    private Matrix inputData;
    private Matrix targetData;

    private Matrix z2;
    private Matrix z3;
    private Matrix yhat;
    private Matrix a2;
    private Matrix dJdW1;
    private Matrix dJdW2;
    private double alpha = 0.1;
    private int iterations = 100000;

    @SuppressWarnings("MismatchedQueryAndUpdateOfCollection")
    private ArrayList<Matrix> initialWeightStorage;

    NeuralNet(){
        boolean testMode = true;
        //noinspection ConstantConditions
        if(!testMode){
             setHyperParameters();
         }else{

             System.out.println("Running in Test mode: \n Default HyperParams set. HyperbolicTangent Activation Function used");

             inputLayerSize = 2;
             outputLayerSize = 1;
             hiddenLayerSize = 3;
         }
         generateWeights();
         generateTrainingMatrices();
         train();
         System.out.println("Estimated values"); yhat.print(1,5);
         System.out.println("Training values"); targetData.print(1, 5);
    }

    private void forward(){
        //TODO: Generalise based on hyperparams

        z2 = new Matrix(inputData.times(weights.get(0)).getArray());

        a2 = activationFunction.apply(z2);

        z3 = new Matrix(a2.times(weights.get(1)).getArray());

        yhat = activationFunction.apply(z3);

    }

    private void backPropagate(Matrix estimated){
        //TODO: Generalise this based on the hyperparams

        //dJdW2
        Matrix m = targetData.minus(estimated).times(-1);
        Matrix delta3 = m.arrayTimes(activationFunction.applyGradFunc(z3));
        dJdW2 = a2.transpose().times(delta3);

        //dJdW1
        Matrix j = delta3.times(weights.get(1).transpose());
        Matrix delta2 = activationFunction.applyGradFunc(z2).arrayTimes(j);
        dJdW1 = inputData.transpose().times(delta2);

        weightCostGradient.set(0, dJdW1);
        weightCostGradient.set(1, dJdW2);


    }

    private void train(){

        System.out.println("Beginning Optimisation (Modified Least Squares)\n Showing every 100th iteration:\n");

        for (int i = 0; i<iterations;i++){
            forward();
            backPropagate(yhat);

            if(i%1000==0){yhat.print(1,3);}

            if(computeCost(yhat)<0.0000001){System.out.println("Optimisation finished terminated by low error at " + i + " iterations: ");break;}

            for(int j = 0; j<weights.size();j++){
                weights.get(j).minusEquals(weightCostGradient.get(j).times(alpha));
            }
           // weights.get(0).minusEquals(dJdW1.times(alpha));
           // weights.get(1).minusEquals(dJdW2.times(alpha));
        }



        //Output
        System.out.println("\nTrained Weights: ");
        weights.get(0).print(1,3);
        weights.get(1).print(1, 3);
        System.out.println("Weight Jacobian's");
        dJdW1.print(1, 10);
        dJdW2.print(1,10);
    }


    private void generateTrainingMatrices(){
        //TODO: Refactor this somehow, it will probably become obsolete
        inputData = normaliseMatrix(new Matrix(trainingIn));
        targetData = normaliseMatrix(new Matrix(trainingOut, 3));
        System.out.println("Normalised Data: ");
        inputData.print(1, 3);
        targetData.print(1, 3);

    }

    /**
     *
     * @param estimates; The cost associated with every training example
     * @return the total mean squares error
     */
    private double computeCost(Matrix estimates){
        Matrix objectiveFunction = targetData.minus(estimates);
        objectiveFunction = objectiveFunction.arrayTimesEquals(targetData.minus(estimates));
        return sum(objectiveFunction)/objectiveFunction.getColumnDimension();
    }

    /**
     * Generate the weights based on the the HyperParameters
     */
    private void generateWeights(){
       //TODO: Make this actually use the hyperparams

       Matrix W1 = newWeightMatrix(inputLayerSize, hiddenLayerSize, inputLayerSize * hiddenLayerNum);
       Matrix W2 = newWeightMatrix(hiddenLayerSize, outputLayerSize, hiddenLayerSize*outputLayerSize);

       weights.add(W1); weights.add(W2);

        for (int i = 0; i < weights.size(); i++) {
            weightCostGradient.add(new Matrix(weights.get(i).getRowDimension(), weights.get(i).getColumnDimension()));
        }

       initialWeightStorage = new ArrayList<>();
       initialWeightStorage.add(W1.copy()); initialWeightStorage.add(W2.copy());

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


    /**
     *
     * @param i Rows
     * @param j Collums
     * @param m number of inputs to the next module
     * @return Weights taken from a distribution of mean = 0 and std dev = m^-1/2
     */
    public static Matrix newWeightMatrix(int i, int j, int m){

        Random r = new Random();
        Matrix weightLayer = new Matrix(i, j);

        for (int k = 0; k < weightLayer.getRowDimension(); k++) {
            for (int l = 0; l < weightLayer.getColumnDimension(); l++) {

                double nextval = r.nextGaussian() * 1/Math.sqrt(m);

                weightLayer.set(k, l, nextval);
            }

        }

    return weightLayer;
    }


    /**
     *
     * @param matrix
     * @return The maximum value in the matrix
     */
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
     * @return The matrix, with the data normalised to 1 based on the maximum value
     */
    public static Matrix normaliseMatrix(Matrix matrix){
        Matrix normalised;
        normalised = matrix.times(1/getMax(matrix));
        return normalised;
    }

    /**
     * Computes the sum the elements of a matrix.
     *
     * @param m the matrix.
     * @return the sum of the elements of the matrix
     */
    public static double sum(Matrix m) {
        int numRows = m.getRowDimension();
        int numCols = m.getColumnDimension();
        double sum = 0;
        // loop through the rows and compute the sum
        for (int i = 0; i < numRows; i++) {
            for (int j = 0; j < numCols; j++) {
                sum += m.get(i, j);
            }
        }
        return sum;
    }

    ///////////////////////////
    /////////Tests/////////////
    ///////////////////////////


    @SuppressWarnings("unused")
    private void testSygmoid(){
        System.out.println("Testing sygmoid: ");
        System.out.println("LogisticFunction(1) = " + activationFunction.apply(1) );
        System.out.println("LogisticFunction(-1, 0 1) = ");
        Matrix out = activationFunction.apply(new Matrix( new double[] {-1, 0, 1}, 1 ));
        out.print(1, 2);

        System.out.println("Testing sygmoidPrime: ");
        System.out.println("Sygmoidprime(1) = " + activationFunction.applyGradFunc(1) );
        System.out.println("LogisticFunction(-1, 0 1) = ");
        Matrix out2 = activationFunction.applyGradFunc(new Matrix( new double[] {-1, 0, 1}, 1 ));
        out2.print(1, 2);


    }



    public static void main(String args[]) {
        @SuppressWarnings("unused") NeuralNet wyvren = new NeuralNet();
    }


}
