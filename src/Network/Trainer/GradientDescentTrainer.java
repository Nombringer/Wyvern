package Network.Trainer;

import Jama.Matrix;
import Network.NeuralNet;

/**
 * Created by fabd on 1/10/17.
 */
public abstract class GradientDescentTrainer extends TrainerImpl {

    protected abstract boolean terminationCondition();
    protected abstract void update();


    GradientDescentTrainer(NeuralNet net, Matrix tIn, Matrix tOut) {
        super(net, tIn, tOut);
    }
    @Override
    public void train() {
        Matrix estimates;

        while(!terminationCondition()){
            estimates = trainingNet.computeEstimates(trainingIn);
            trainingNet.backProp(estimates, trainingOut);
            update();
        }
    }

}
