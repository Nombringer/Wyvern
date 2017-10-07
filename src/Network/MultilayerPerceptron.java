package Network;

import Jama.Matrix;
import Network.ActivationFunction.ActivationFunction;
import Network.ActivationFunction.HyperbolicTangent;
import Network.Layer.BiasLayer;
import Network.Layer.WeightLayer;
import Network.Trainer.UpdateRule;

import java.util.ArrayList;
import static Ultil.MatrixUtils.sum;

/**
 * Base Neural Net class for project Wyvern.
 * All done using basic linear algebra and matrix operands from the Jama library
 */



public class MultilayerPerceptron implements NeuralNet { //TODO: Make abstract


    //Stores the sizes of each layer. including input and output. Matrices are generated based off this.
    private ArrayList<Integer> layerSizes = new ArrayList<>(2);

    //Default activation function for all bias layers in the network
    private ActivationFunction BiasLayerFunction = new HyperbolicTangent();

    //Storage for Bias and Weight classes
    private ArrayList<Matrix> weights = new ArrayList<>();
    private ArrayList<Matrix> weightCostGradient = new ArrayList<>();
    private ArrayList<BiasLayer> biasLayers = new ArrayList<>();

    private Matrix inputData;
    private Matrix targetData;
    private Matrix estimates;


    public MultilayerPerceptron(Matrix in, Matrix out, ArrayList<Integer> hiddenLayerSizes){
        setInputData(in);
        for (Integer i: hiddenLayerSizes){
            layerSizes.add(i);
        }
        setOutData(out);
        generateWeightLayers();
        generateBiasLayers(inputData);
    }

    ////////////////////////////////////
    /////FORWARD AND BACK PROPAGATION///
    ////////////////////////////////////

    /**
     * Propagates an input through the matrix
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

    /**
     * @param in, input matrix, needs match input size of training set
     * @return double array of the estimates
     */
    public double[][] getEstimates(Matrix in){
        generateBiasLayers(in);
        forwardProp(in);
        estimates.print(1,3);
        return estimates.getArray();
    }

    /**
     * @param in input Matrix, needs to match input size of training set.
     * @return
     */
    public Matrix computeEstimates(Matrix in){
        generateBiasLayers(in);
        forwardProp(in);
        return estimates.copy();
    }


    /////////////////////
    //////TRAINING///////
    /////////////////////



    /**
     * Updates every weight in the net based on the current applyUpdateRule rule
     */
    public void applyUpdateRule(UpdateRule rule ){
        rule.update(weights, weightCostGradient);
    }

    /**
     * @param estimates; The cost associated with every training example
     * @return the total mean squares error
     */
    public double computeCost(Matrix estimates){
        Matrix objectiveFunction = targetData.copy().minus(estimates.copy());
        objectiveFunction = objectiveFunction.arrayTimesEquals(targetData.copy().minus(estimates.copy()));
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
        biasLayers = new ArrayList<>();
        for (int i = 1; i <(layerSizes.size()); i++) {
            biasLayers.add(new BiasLayer(input.getRowDimension(), layerSizes.get(i) , BiasLayerFunction));
        }
    }

    ///////////////////////////
    //////HELPER FUNCTIONS/////
    ///////////////////////////

    private void setInputData(Matrix trainingIn){
        inputData = trainingIn;
        layerSizes.add(trainingIn.getColumnDimension());
    }

    private void setOutData(Matrix trainingOut) {
        targetData = trainingOut;
        layerSizes.add(trainingOut.getColumnDimension());
    }

    @Override
    public void printCurrentCost(){
        System.out.println("Cost of current estimates :");
        System.out.println(computeCost(estimates));
    }

    @Override
    public double getCost() {
        return computeCost(estimates);
    }
}
