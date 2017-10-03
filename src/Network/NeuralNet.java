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

    void applyUpdateRule(UpdateRule rule);

    double[][] getEstimates(Matrix input);

    Matrix computeEstimates(Matrix in);

    double computeCost(Matrix estimates);

    double getCost();

    void printCurrentCost();
}
