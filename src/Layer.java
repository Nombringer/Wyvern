import Jama.Matrix;

/**
 * Created by fabd on 12/09/17.
 */
public class Layer extends Matrix {

    private ActivationFunction layerActivationFunction;
    private int neuronNum;

    public ActivationFunction getLayerActivationFunction() {
        return layerActivationFunction;
    }

    public void setLayerActivationFunction(ActivationFunction layerActivationFunction) {
        this.layerActivationFunction = layerActivationFunction;
    }

    public int getNeuronNum() {
        return neuronNum;
    }


    public Layer(int i, int i1) {
        super(i, i1);
        neuronNum = this.getColumnDimension();
    }

    public Layer(int i, int i1, double v) {
        super(i, i1, v);
        neuronNum = this.getColumnDimension();
    }

    public Layer(double[][] doubles) {
        super(doubles);
        neuronNum = this.getColumnDimension();
    }

    public Layer(double[][] doubles, int i, int i1) {
        super(doubles, i, i1);
        neuronNum = this.getColumnDimension();
    }

    public Layer(double[] doubles, int i) {
        super(doubles, i);
        neuronNum = this.getColumnDimension();
    }

    private class Neuron {


        private ActivationFunction activationFunction;

        private ActivationFunction getActivationFunction() {
            return activationFunction;
        }

        private void setActivationFunction(ActivationFunction activationFunction) {
            this.activationFunction = activationFunction;
        }
    }

}
