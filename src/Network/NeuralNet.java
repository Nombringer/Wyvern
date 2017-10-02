package Network;

import Jama.Matrix;

/**
 * Created by fabd on 28/09/17.
 */
public interface NeuralNet {

    void forwardProp(Matrix input);

    void backProp(Matrix estimates);

    double[][] getEstimates(Matrix input);

    void update(); //TODO: Should pass in a lambda to this.

    double computeCost(Matrix estimates);

    Matrix computeEstimates(Matrix in);
}
