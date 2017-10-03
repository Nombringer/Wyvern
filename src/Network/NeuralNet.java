package Network;

import Jama.Matrix;
import Network.Trainer.UpdateRule;

import java.util.ArrayList;

/**
 * Created by fabd on 28/09/17.
 */
public interface NeuralNet {

    void forwardProp(Matrix input);

    void backProp(Matrix estimates);

    double[][] getEstimates(Matrix input);

    void applyUpdateRule(UpdateRule rule); //TODO: Should pass in a lambda to this.

    double computeCost(Matrix estimates);

    Matrix computeEstimates(Matrix in);

    double getCost();

    void printCurrentCost();
}
