package Trainer;

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
    public void Train(NeuralNet net, Matrix trainingInputs, Matrix trainingOutputs) {

        while(!terminationCondition()){
            Matrix estimates = net.computeEstimates(trainingInputs);
            net.backProp(estimates);
            update();
        }
    }

    protected abstract boolean terminationCondition();
    protected abstract void update();

}
