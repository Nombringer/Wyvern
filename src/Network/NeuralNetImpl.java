package Network;

import Jama.Matrix;
import Network.ActivationFunction.ActivationFunction;
import Network.ActivationFunction.HyperbolicTangent;
import Network.Layer.BiasLayer;
import Network.Layer.WeightLayer;
import Trainer.Trainer;

import java.util.ArrayList;
import static Ultil.MatrixUtils.normaliseMatrix;
import static Ultil.MatrixUtils.sum;

/**
 * Base Neural Net class for project Wyvern.
 * All done using basic linear algebra and matrix operands from the Jama library
 */


/////////////////////////////////////////////
/////NEXT THINGS, CHANGES AND REFACTORS//////
/////////////////////////////////////////////


@SuppressWarnings({"FieldCanBeLocal", "WeakerAccess", "JavaDoc"})
public class NeuralNetImpl implements NeuralNet { //TODO: Make abstract


    //Stores the sizes of each layer. including input and output. Matrices are generated based off this.
    private ArrayList<Integer> layerSizes = new ArrayList<>(2);

    //Default activation function for all bias layers in the network
    private ActivationFunction BiasLayerFunction = new HyperbolicTangent();

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


    //Example training set from WelchLabs
    private final boolean testMode = true;
    private final double[][] trainingIn =  {{3, 5}, {5, 1}, {10, 2}};
    private final double[]   trainingOut = {75, 82, 93};

    //Arbitrary thresholds because nothing better has been implemented yet
    private final int maxIterations = 100000;
    private final double costThreshold = 0.0000001;
    private final double learningRate = 0.1;



    public NeuralNetImpl(){

        //////////////////////
        /////TESTING CRAP/////
        //////////////////////


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

            System.out.println("Testing input [3][5]");
            double[] testIn = {3, 5}; Matrix input = new Matrix(testIn ,1);
            System.out.println(getEstimates(input));

        }

    }

    public NeuralNetImpl(Matrix in, Matrix out, ArrayList<Integer> hiddenLayerSizes){



    }




    ////////////////////////////////////
    /////FORWARD AND BACK PROPAGATION///
    ////////////////////////////////////

    /**
     * Propgates an input through the matrix
     * @param input
     */
    public void forwardProp(Matrix input){
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

    public void backProp(Matrix estimated){
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

    public double[][] getEstimates(Matrix in){
        generateBiasLayers(in);
        forwardProp(in);
        estimates.print(1,3);
        return estimates.getArray();
    }

    /////////////////////
    //////TRAINING///////
    /////////////////////


    /**
     * Trains the network, using the specified:
     * Update Rule
     * Termination Rule
     * Activation Function
     */
    private void train(){
        int iterations = 0;
        backProp(estimates);
        System.out.println("Beginning Optimisation (Modified Least Squares)\n Showing every 1000th iteration:\n");
        while (checkEstimates()&&iterations<maxIterations){
            forwardProp(inputData);
            backProp(estimates);
            //if(iterations%10000 ==0){ printWeightCostGradient(0);}
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
    public void update(){
        //TODO: Refactor into an "Update Rule" possibly in a trainer package.
        for(int j = 0; j<weights.size();j++){
            weights.get(j).minusEquals(weightCostGradient.get(j).times(learningRate));
        }
    }

    /**
     * @param estimates; The cost associated with every training example
     * @return the total mean squares error
     */
    public double computeCost(Matrix estimates){
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
           WeightLayer weightLayer = new WeightLayer(layerSizes.get(i), layerSizes.get(i+1), BiasLayerFunction, layerSizes.get(i)*layerSizes.get(i+1));
           weights.add(weightLayer);
       }
       for (Matrix weight : weights) {
            weightCostGradient.add(new WeightLayer(weight.getRowDimension(), weight.getColumnDimension(), BiasLayerFunction,0 ));
       }
    }

    /**
     * @param input
     * Generates the underlying matrices for the bias layers.
     * Dimensions are based on the dimensions of the input matrix, ie # of training examples
     */
    private void generateBiasLayers(Matrix input){
        //TODO: Currently the layers are regenerated on input, meaning we can edit individual activation functions, maybe add some kind of thing that lets you scale the size
        biasLayers = new ArrayList<>();
        for (int i = 1; i <(layerSizes.size()); i++) {
            biasLayers.add(new BiasLayer(input.getRowDimension(), layerSizes.get(i) , BiasLayerFunction));
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

    public void setInputData(Matrix trainingIn){
        inputData = normaliseMatrix(trainingIn);
        layerSizes.set(0, trainingIn.getColumnDimension());
    }

    public void setOutData(Matrix trainingOut) {
        targetData = normaliseMatrix(trainingOut);
        layerSizes.set(layerSizes.size() -1, trainingOut.getColumnDimension());
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
        System.out.println("Network.ActivationFunctionImpl.LogisticFunctionImpl(1) = " + BiasLayerFunction.apply(1) );
        System.out.println("Network.ActivationFunctionImpl.LogisticFunctionImpl(-1, 0 1) = ");
        Matrix out = BiasLayerFunction.apply(new Matrix( new double[] {-1, 0, 1}, 1 ));
        out.print(1, 2);

        System.out.println("Testing sygmoidPrime: ");
        System.out.println("Sygmoidprime(1) = " + BiasLayerFunction.applyGradFunc(1) );
        System.out.println("Network.ActivationFunctionImpl.LogisticFunctionImpl(-1, 0 1) = ");
        Matrix out2 = BiasLayerFunction.applyGradFunc(new Matrix( new double[] {-1, 0, 1}, 1 ));
        out2.print(1, 2);


    }



    public static void main(String args[]) {
        @SuppressWarnings("unused") NeuralNetImpl wyvren = new NeuralNetImpl();
    }


}
