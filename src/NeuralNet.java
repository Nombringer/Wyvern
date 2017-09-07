import Jama.Matrix;

import java.util.ArrayList;
import java.util.Random;

/**
 * Base Neural Net class for project Wyvern.
 */

//TODO: Implement early stopping and regularisation
@SuppressWarnings({"FieldCanBeLocal", "WeakerAccess", "JavaDoc"})
public class NeuralNet {
    private boolean testMode = true;

    private final double[][] trainingIn =  {{3, 5}, {5, 1}, {10, 2}};
    private final double[]   trainingOut = {75, 82, 93};

    private final int maxIterations = 100000;
    private final double costThreshold = 0.0000001;

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

    @SuppressWarnings("MismatchedQueryAndUpdateOfCollection")
    private ArrayList<Matrix> initialWeightStorage;

    NeuralNet(){
         generateTrainingMatrices();
         setHyperParameters();
         generateWeights();
         train();
         System.out.println("Estimated values"); yhat.print(1,5);
         System.out.println("Training values"); targetData.print(1, 5);
    }

    private void forwardPropagate(){
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
        int iterations = 1;

        //TODO: This is an ugly way to avoid nullpointers
        forwardPropagate();
        backPropagate(yhat);
        System.out.println("Beginning Optimisation (Modified Least Squares)\n Showing every 1000th iteration:\n");

        while (checkEstimates()&&iterations<maxIterations){
            forwardPropagate();
            backPropagate(yhat);

            if (iterations%1000 ==0){printWeights(0); printCurrentCost();}

            update();
            iterations++;
        }

        //Output
        System.out.println("\nTrained Weights: ");
        printWeights(0);
    }


    private void generateTrainingMatrices(){
        //TODO: Refactor this somehow, it will probably become obsolete
        inputData = normaliseMatrix(new Matrix(trainingIn));
        targetData = normaliseMatrix(new Matrix(trainingOut, 3));
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
       Matrix W2 = newWeightMatrix(hiddenLayerSize, outputLayerSize, hiddenLayerSize * outputLayerSize);

       weights.add(W1); weights.add(W2);

        for (Matrix weight : weights) {
            weightCostGradient.add(new Matrix(weight.getRowDimension(), weight.getColumnDimension()));
        }

       initialWeightStorage = new ArrayList<>();
       initialWeightStorage.add(W1.copy()); initialWeightStorage.add(W2.copy());

    }


    /**
     * Set the HyperParameters based on console input
     */
    private void setHyperParameters(){
        if(testMode){
            inputLayerSize = inputData.getColumnDimension();
            hiddenLayerSize = 2;
            outputLayerSize = targetData.getColumnDimension();
            return;
        }

        System.out.println("Setting HyperParameters: \n Please enter the input layer size:");
        inputLayerSize = Integer.parseInt(System.console().readLine());

        System.out.println("Please enter the outputLayer size");
        outputLayerSize = Integer.parseInt(System.console().readLine());

        System.out.println("Please enter the Hidden Layer size");
        hiddenLayerSize = Integer.parseInt(System.console().readLine());


    }

    /**
     *
     * @return true while the cost of the estimates is greater than the threshold value
     */
    public boolean checkEstimates(){
        if(computeCost(yhat)<costThreshold){
            System.out.println("Optimisation finished, cost in required threshold");
            return false;
        }
        return true;
    }


    /**
     * Updates every weight in the net based on the current update rule
     */
    private void update(){

        //TODO: Implement multiple update rule options

        for(int j = 0; j<weights.size();j++){
            weights.get(j).minusEquals(weightCostGradient.get(j).times(alpha));
        }
    }


    ///////////////////////////
    //////HELPER FUNCTIONS/////
    ///////////////////////////


    /**
     *
     * @param i Rows
     * @param j Columns
     * @param m number of inputs to the next module
     * @return Weights taken from a distribution of mean = 0 and std dev = m^-1/2
     */
    public static Matrix newWeightMatrix(int i, int j, int m){

        Random r = new Random();
        Matrix weightLayer = new Matrix(i, j);

        for (int k = 0; k < weightLayer.getRowDimension(); k++) {
            for (int l = 0; l < weightLayer.getColumnDimension(); l++) {
                double nextVal;

                nextVal = r.nextGaussian() * 1/Math.sqrt(m);
                weightLayer.set(k, l, nextVal);
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

    ////////////////////////////////////
    ////////Getters/Setters/Printers////
    ////////////////////////////////////



    @SuppressWarnings("unused")
    public void printNormalisedData(){
        System.out.println("Normalised Data: ");
        inputData.print(1, 3);
        targetData.print(1, 3);
    }

    @SuppressWarnings("unused")
    public void printTrainingOutputs() {
        System.out.println("Training outputs:  ");
        targetData.print(1, 3);
    }

    public void printCurrentCost(){
        System.out.println("Cost of current estimates :");
        System.out.println(computeCost(yhat));
    }

    @SuppressWarnings("unused")
    public void printWeightCostGradient(int layer) {
        assert layer<=hiddenLayerNum&&layer>-1:"Layer does not exist";

        if(layer==0){
            System.out.println("Cost Gradient of current weight layers");
            for(Matrix m : weightCostGradient){
                m.print(1,3);}
        }else{
            System.out.println("Current cost gradient of weights at layer " + layer);
            weightCostGradient.get(layer).print(1, 3);}

    }

    public void printWeights(@SuppressWarnings("SameParameterValue") int layer) {
     assert layer<=hiddenLayerNum&&layer>-1:"Layer does not exist";

     if(layer==0){
         System.out.println("Current weights");
         for(Matrix m : weights){
                m.print(1,3);}
        }else{
         System.out.println("Current weights at layer " + layer);
         weights.get(layer).print(1, 3);}

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
