package Network.Trainer;

import Jama.Matrix;
import Network.NeuralNet;


/**
 * Created by fabd on 7/10/17.
 */
public class SGDTrainer extends GradientDescentTrainer {

    private int iterations;
    private int pos;


    SGDTrainer(NeuralNet net, Matrix tIn, Matrix tOut) {
        super(net, tIn, tOut);
        pos = 0;
    }

    @Override
    protected boolean terminationCondition() {
        return false;
    }

    @Override
    protected void update() {

    }

}
