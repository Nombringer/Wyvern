package Network.Layer;

import Jama.Matrix;
import Network.ActivationFunction.ActivationFunction;
import Network.ActivationFunction.ActivationFunctionImpl;

/**
 * Created by fabd on 12/09/17.
 */
  abstract class Layer extends Matrix {

    protected ActivationFunction layerActivationFunction;

    public Layer(int i, int i1, ActivationFunction layerFunction) {
        super(i, i1);
        layerActivationFunction = layerFunction;
    }


    public void applyFunction(ActivationFunctionImpl function){
        for (int i = 0; i < this.getRowDimension(); i++) {

            for (int j = 0; j < this.getColumnDimension(); j++) {
                this.set(i, j, function.apply((this.get(i,j))));
            }
        }
    }

    public void applyLayerFunction(){
        for (int i = 0; i < this.getRowDimension(); i++) {
            for (int j = 0; j < this.getColumnDimension(); j++) {
                this.set(i, j, layerActivationFunction.apply((this.get(i,j))));
            }
        }
    }


    public void matrixTimesEquals(Matrix A, Matrix B){
        Matrix vals = A.times(B);
        for (int i = 0; i < this.getRowDimension(); i++) {
            for (int j = 0; j < this.getColumnDimension(); j++) {
                this.set(i, j, vals.get(i, j));
            }
        }
    }

    //////////////////////////////
    /////GETTERS//AND//SETTERS////
    //////////////////////////////

    public void setLayerActivationFunction(ActivationFunction layerActivationFunction) {
        this.layerActivationFunction = layerActivationFunction;
    }

    public ActivationFunction getLayerActivationFunction() {
        return layerActivationFunction;
    }


}
