package Network.Layer;

import Jama.Matrix;
import Network.ActivationFunction.ActivationFunction;

/**
 * Created by fabd on 12/09/17.
 */
public class BiasLayer extends Layer {

    private Neuron[] neurons;
    private Matrix rawVals;

    public BiasLayer(int i, int i1, ActivationFunction layerFunction) {
        super(i, i1, layerFunction);
        neurons = new Neuron[this.getColumnDimension()];

        for (int j = 0; j < neurons.length; j++) {
            neurons[j] = new Neuron(i, layerFunction);
        }
    }



    public void activate(){
        rawVals = new Matrix(this.getArray());
        for (int i = 0; i < this.getRowDimension(); i++) {
            for (int j = 0; j < this.getColumnDimension(); j++) {
                this.set(i, j, neurons[j].getNeuronFunction().apply(this.get(i, j)));
            }
        }
    }

    public Matrix getGradients(){
        Matrix gradients = layerActivationFunction.applyGradFunc(rawVals.copy());
        return gradients;
    }

    public void setActivationFunction(int index, ActivationFunction func){
        neurons[index].setNeuronFunction(func);
    }

    public ActivationFunction getActivationFunction(int index){
        return neurons[index].getNeuronFunction();
    }

    //TODO: Maybe this implementation is a little ugly. Find a way to make it work or just din't use the inner class.
    private class Neuron {
        private ActivationFunction neuronFunction;

        Neuron(int pos, ActivationFunction function){
            neuronFunction = function;
        }

        ActivationFunction getNeuronFunction() {
            return neuronFunction;}
        void setNeuronFunction(ActivationFunction neuronFunction) {
            this.neuronFunction = neuronFunction;}
    }

}
