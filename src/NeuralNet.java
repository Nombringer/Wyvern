import Jama.Matrix;
import java.util.ArrayList;
import java.util.Random;

/**
 * Base Neural Net class for project Wyvern.
 * All done using basic linear algebra and matrix operands from the Jama library
 */


/////////////////////////////////////////////
/////NEXT THINGS, CHANGES AND REFACTORS//////
/////////////////////////////////////////////

//TODO: Figure out package structure.
//TODO: Figure out the best way to implement better training methods.
//TODO: Load up the MINST dataset and have some fun.
//TODO: Write Utils package.


@SuppressWarnings({"FieldCanBeLocal", "WeakerAccess", "JavaDoc"})
public class NeuralNet {

    //Example training set from WelchLabs
    private final boolean testMode = true;
    private final double[][] trainingIn =  {{3, 5}, {5, 1}, {10, 2}};
    private final double[]   trainingOut = {75, 82, 93};

    //Arbitrary thresholds because nothing better has been implemented yet
    private final int maxIterations = 100000;
    private final double costThreshold = 0.0000001;
    private final double learningRate = 0.1;

    //Stores the sizes of each layer. including input and output. Matrices are generated based off this.
    private ArrayList<Integer> layerSizes = new ArrayList<>();

    //Default activation function for all bias layers in the network
    private ActivationFunction activationFunction = new HyperbolicTangent();

    //Storage for Bias and Weight classes
    private ArrayList<Matrix> weights = new ArrayList<>();
    private ArrayList<Matrix> weightCostGradient = new ArrayList<>();
    private ArrayList<BiasLayer> biasLayers = new ArrayList<>();



    //TODO: This stuff might not make so much now that sizes are modular and stored elsewhere, refactor.
    private int inputLayerSize;
    private int outputLayerSize;
    private Matrix inputData;
    private Matrix targetData;
    private Matrix estimates;


    NeuralNet(){
        //TODO: Figure out what should actually be here.

        if(testMode){
            //Generates and trains the network on the WelchLabs training set.
            generateTrainingMatrices();
            inputLayerSize = inputData.getColumnDimension();
            outputLayerSize = targetData.getColumnDimension();
            layerSizes.add(inputLayerSize); layerSizes.add(10); layerSizes.add(outputLayerSize);

            generateWeightLayers();
            generateBiasLayers(inputData);
            forwardProp(inputData);
            train();
            System.out.println("Estimated values"); estimates.print(1,5);
            System.out.println("Training values"); targetData.print(1, 5);
        }

    }

    ////////////////////////////////////
    /////FORWARD AND BACK PROPAGATION///
    ////////////////////////////////////

    /**
     * Propgates an input through the matrix
     * @param input
     */
    private void forwardProp(Matrix input){
        //TODO: Start adding size exceptions

        Matrix previous = input;
        int i = 0;

        for(BiasLayer biasLayer: biasLayers){
            biasLayer.matrixTimesEquals(previous, weights.get(i));
            biasLayer.activate();
            previous = biasLayer;
            i++;
        }
        estimates = biasLayers.get(biasLayers.size()-1);
    }

    private void backProp(Matrix estimated){
        Matrix dJdW;
        forwardProp(inputData);

        //Compute the gradient of the last layer.
        Matrix m = targetData.minus(estimated).times(-1);
        Matrix delta = m.arrayTimes(biasLayers.get(biasLayers.size() -1).getGradients());
        dJdW = biasLayers.get(biasLayers.size() -2).transpose().times(delta);
        weightCostGradient.set(weightCostGradient.size() -1, dJdW);

        //Use the first gradients to backpropagate the error back through the rest of the network.
        for (int i = 1; i<weights.size(); i++){
            Matrix matrix = delta.times(weights.get(weights.size() -i ).transpose());
            Matrix deltaPrime = biasLayers.get(biasLayers.size() -i -1).arrayTimes(matrix);

            if (biasLayers.size() -i -1 == 0){
             dJdW = inputData.transpose().times(deltaPrime);}else{
                dJdW = biasLayers.get(biasLayers.size() -i -2).transpose().times(deltaPrime);
            }
            weightCostGradient.set(weightCostGradient.size() -i -1, dJdW);
            delta = deltaPrime;
        }


    }

    /////////////////////
    //////TRAINING///////
    /////////////////////


    /**
     * Trains the network, using the specified:
     * Update Rule
     * Termination Rule
     * Activation Function
     * TODO: The above are not fully implemented yet, there is a simple MaxIterations constant and a flat threshold for deciding if training should finish.
     */
    private void train(){
        int iterations = 0;
        backProp(estimates);
        System.out.println("Beginning Optimisation (Modified Least Squares)\n Showing every 1000th iteration:\n");
        while (checkEstimates()&&iterations<maxIterations){
            forwardProp(inputData);
            backProp(estimates);
            if(iterations%10000 ==0){ printWeightCostGradient(0);}
            update();
            iterations++;
        }

    }

    /**
     * @return true while the cost of the estimates is greater than the threshold value
     */
    public boolean checkEstimates(){
        if(computeCost(estimates)<costThreshold){
            System.out.println("Optimisation finished, cost in required threshold");
            return false;
        }
        return true;
    }


    /**
     * Updates every weight in the net based on the current update rule
     */
    private void update(){
        //TODO: Refactor into an "Update Rule" possibly in a trainer package.
        for(int j = 0; j<weights.size();j++){
            weights.get(j).minusEquals(weightCostGradient.get(j).times(learningRate));
        }
    }

    /**
     * @param estimates; The cost associated with every training example
     * @return the total mean squares error
     */
    private double computeCost(Matrix estimates){
        Matrix objectiveFunction = targetData.minus(estimates);
        objectiveFunction = objectiveFunction.arrayTimesEquals(targetData.minus(estimates));
        return sum(objectiveFunction)/objectiveFunction.getColumnDimension();
    }


    //////////////////////////////////////////////////
    //////LAYER GENERATION AND AND PARAMETRISATION////
    //////////////////////////////////////////////////

    /**
     * Generate the weights based on the the HyperParameters
     * Also populates a list which corresponds to the derivatives of each layer with respect to the cost.
     */
    private void generateWeightLayers(){
       for(int i = 0; i<layerSizes.size() -1; i++){
           WeightLayer weightLayer = new WeightLayer(layerSizes.get(i), layerSizes.get(i+1), activationFunction, layerSizes.get(i)*layerSizes.get(i+1));
           weights.add(weightLayer);
       }
       for (Matrix weight : weights) {
            weightCostGradient.add(new WeightLayer(weight.getRowDimension(), weight.getColumnDimension(), activationFunction,0 ));
       }
    }

    /**
     * @param input
     * Generates the underlying matrices for the bias layers.
     * Dimensions are based on the dimensions of the input matrix, ie # of training examples
     */
    private void generateBiasLayers(Matrix input){
        for (int i = 1; i <(layerSizes.size()); i++) {
            biasLayers.add(new BiasLayer(input.getRowDimension(), layerSizes.get(i) , activationFunction));
        }
    }

    private void generateTrainingMatrices(){
        //TODO: This should be refactored into utilities
        inputData = normaliseMatrix(new Matrix(trainingIn));
        targetData = normaliseMatrix(new Matrix(trainingOut, 3));
    }



    ///////////////////////////
    //////HELPER FUNCTIONS/////
    ///////////////////////////

    //TODO: Migrate to Utils if needed.

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
        System.out.println(computeCost(estimates));
    }

    @SuppressWarnings("unused")
    public void printWeightCostGradient(int layer) {
        if(layer==0){
            System.out.println("Cost Gradient of current weight biasLayers");
            for(Matrix m : weightCostGradient){
                m.print(1,6);}
        }else{
            System.out.println("Current cost gradient of weights at layer " + layer);
            weightCostGradient.get(layer).print(1, 3);}

    }

    public void printWeights(@SuppressWarnings("SameParameterValue") int layer) {
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

    //TODO: Make some. This is mildly embarrassing.
    //TODO: Benchmarking.

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
