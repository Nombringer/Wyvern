package Network;

import Network.ActivationFunction.ActivationFunction;
import Network.ActivationFunction.ActivationFunctionImpl;

import java.util.Random;

/**
 * Created by fabd on 18/09/17.
 */
public class WeightLayer extends Layer {
    private int inputNum;

    public WeightLayer(int i, int i1, ActivationFunction layerFunction, int inputnum) {
        super(i, i1, layerFunction);
        Random r = new Random();
        if(inputnum ==0){inputNum = getRowDimension();}else{inputNum = inputnum;}

        for (int k = 0; k < this.getRowDimension(); k++) {
            for (int l = 0; l < this.getColumnDimension(); l++) {

                double nextVal;
                nextVal = r.nextGaussian() * 1/Math.sqrt(inputNum);
                this.set(k, l, nextVal);
            }
        }

    }

    public WeightLayer copy(){
        WeightLayer newWLayer = new WeightLayer(this.getRowDimension(), this.getColumnDimension(), this.getLayerActivationFunction(), inputNum);
        for (int i = 0; i < this.getRowDimension(); i++) {
            for (int j = 0; j < this.getColumnDimension(); j++) {
                newWLayer.set(i, j, this.get(i, j));
            }

        }
        return newWLayer;
    }


}
