package Network.Trainer;

import Jama.Matrix;
import Network.NeuralNet;





/**
 * Created by fabd on 7/10/17.
 */
public class SGDTrainer extends GradientDescentTrainer {

    private int epochlimit; //TODO: Something better
    private int miniBatchSize;
    int pos;



    SGDTrainer(NeuralNet net, Matrix tIn, Matrix tOut) {
        super(net, tIn, tOut);

        Matrix fullIn = trainingIn;
        Matrix fullOut = trainingOut;
    }

    @Override
    protected boolean terminationCondition() {
        return false;
    }

    @Override
    protected void update() {

    }

}
