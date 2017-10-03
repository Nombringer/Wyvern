package Network.Trainer;

import Jama.Matrix;
import Network.NeuralNet;

/**
 * Created by fabd on 1/10/17.
 */
public abstract class GradientDescentTrainer extends TrainerImpl {


    GradientDescentTrainer(NeuralNet net, Matrix tIn, Matrix tOut) {
        super(net, tIn, tOut);
    }

    @Override
    public void train() {

        while(!terminationCondition()){
            Matrix estimates = trainingNet.computeEstimates(trainingIn);
            trainingNet.backProp(estimates);
            update();
        }
    }

    protected abstract boolean terminationCondition();
    protected abstract void update();

}
