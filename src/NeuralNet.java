import Jama.Matrix;

import java.util.ArrayList;
import java.util.Random;

/**
 * Base Neural Net class for project Wyvern.
 * All done using basic linear algebra and matrix operands from the Jama library
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
    private int hiddenLayerSize = 3;

    private ArrayList<Integer> layerSizes = new ArrayList<>();

    private ActivationFunction activationFunction = new HyperbolicTangent();
    private ArrayList<Matrix> weights = new ArrayList<>();
    private ArrayList<Matrix> weightCostGradient = new ArrayList<>(2);
    private ArrayList<BiasLayer> biasLayers = new ArrayList<>();
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
        if(testMode){
            inputLayerSize = inputData.getColumnDimension();
            outputLayerSize = targetData.getColumnDimension();
            layerSizes.add(2); layerSizes.add(3); layerSizes.add(1);
        }
         setHyperParameters();
         generateWeightLayers();
         inputData.print(1, 3);
         generateBiasLayers(inputData);
         forwardProp(inputData);
         train();
         System.out.println("Estimated values"); yhat.print(1,5);
         System.out.println("Training values"); targetData.print(1, 5);
    }

    private void forwardPropagate(Matrix input){
        //TODO: Generalise based on hyperparams

        z2 = new Matrix(input.times(weights.get(0)).getArray());

        a2 = activationFunction.apply(z2);

        z3 = new Matrix(a2.times(weights.get(1)).getArray());

        yhat = activationFunction.apply(z3);

    }

    private void forwardProp(Matrix input){

        Matrix previous = input;
        int i = 0;

        for(BiasLayer biasLayer: biasLayers){
            biasLayer.matrixTimesEquals(previous, weights.get(i));
            biasLayer.activate();
            previous = biasLayer;
            i++;
        }
        yhat = biasLayers.get(biasLayers.size()-1);
    }

    private void backPropagate(Matrix estimated){
        //TODO: Generalise this based on the hyperparams
        forwardPropagate(inputData);

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

    private void backProp(Matrix estimated){

        //TODO: Generalise this based on the hyperparams
        forwardProp(inputData);

        //dJdW2
        Matrix m = targetData.minus(estimated).times(-1);
        Matrix delta3 = m.arrayTimes(biasLayers.get(biasLayers.size() -1).getGradients());
        dJdW2 = biasLayers.get(biasLayers.size() -2).transpose().times(delta3);

        //dJdW1
        Matrix j = delta3.times(weights.get(1).transpose());
        Matrix delta2 = biasLayers.get(biasLayers.size() -2).arrayTimes(j);
        dJdW1 = inputData.transpose().times(delta2);

        weightCostGradient.set(0, dJdW1);
        weightCostGradient.set(1, dJdW2);


        Matrix deltaPrev = new Matrix(1, 1);
        Matrix matrix;

        for (int i = biasLayers.size()-1; i >=0; i--) {
            if(i == biasLayers.size()-1){ matrix = targetData.minus(estimated).times(-1);}

            else{matrix = deltaPrev.times(weights.get(i).transpose());}
            Matrix delta = matrix.arrayTimes(biasLayers.get(i).getGradients());
            Matrix weightLayerGradient = biasLayers.get(i-1).transpose().times(delta);
            deltaPrev = delta;
            weightCostGradient.set(i, weightLayerGradient);
        }



    }

    /**
     * Trains the network, using the specified:
     * Update Rule
     * Termination Rule
     * Activation Function
     * TODO: Parametrise these somehow
     */
    private void train(){
        int iterations = 0;
        backProp(yhat);
        System.out.println("Beginning Optimisation (Modified Least Squares)\n Showing every 1000th iteration:\n");
        while (checkEstimates()&&iterations<maxIterations){
            forwardProp(inputData);
            backProp(yhat);
            //if(iterations%10000 ==0){ printCurrentCost();}
            update();
            iterations++;
        }

    }

    private void traint(){
        int iterations = 0;
        backPropagate(yhat);
        System.out.println("Beginning Optimisation (Modified Least Squares)\n Showing every 1000th iteration:\n");
        while (checkEstimates()&&iterations<maxIterations){
            forwardPropagate(inputData);
            backPropagate(yhat);
            if(iterations%1000 ==0){printWeights(0); printCurrentCost();}
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

    private void generateLayers(Matrix input){
        layerSizes.add(input.getRowDimension());
    }

    /**
     * Generate the weights based on the the HyperParameters
     */
    private void generateWeightLayers(){
       //TODO: Make this actually use the hyperparams

       WeightLayer W1 = new WeightLayer(inputLayerSize, hiddenLayerSize, activationFunction, inputLayerSize*hiddenLayerNum);
       WeightLayer W2 = new WeightLayer(hiddenLayerSize, outputLayerSize, activationFunction,  hiddenLayerSize*outputLayerSize);


        weights.add(W1); weights.add(W2);

        for (Matrix weight : weights) {
            weightCostGradient.add(new WeightLayer(weight.getRowDimension(), weight.getColumnDimension(), activationFunction,0 ));
        }

       initialWeightStorage = new ArrayList<>();
       initialWeightStorage.add(W1.copy()); initialWeightStorage.add(W2.copy());

    }

    private void generateBiasLayers(Matrix input){
        for (int i = 1; i <(layerSizes.size()); i++) {
            biasLayers.add(new BiasLayer(input.getRowDimension(), layerSizes.get(i) , activationFunction));
        }
    }


    /**
     * Set the HyperParameters based on console input
     */
    private void setHyperParameters(){


        //TODO: Start getting this to work.
        /*
        System.out.println("Setting HyperParameters: \n Please enter the input layer size:");
        inputLayerSize = Integer.parseInt(System.console().readLine());

        System.out.println("Please enter the outputLayer size");
        outputLayerSize = Integer.parseInt(System.console().readLine());

        System.out.println("Please enter the Hidden Layer size");
        hiddenLayerSize = Integer.parseInt(System.console().readLine());
        */

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
            System.out.println("Cost Gradient of current weight biasLayers");
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
