package Trainer;

import Jama.Matrix;
import Network.NeuralNet;

/**
 * Created by fabd on 3/10/17.
 */
public class NaiveGradientDescent extends GradientDescentTrainer {

    private int iterations;
    private int iterationThreshold;
    private int gradientTheshhold;

    NaiveGradientDescent(NeuralNet net, Matrix tIn, Matrix tOut) {
        super(net, tIn, tOut);
    }

    @Override
    protected boolean terminationCondition() {
        if(iterations>=iterationThreshold){
            return true;
        }
        return false;
    }


    //   for(int j = 0; j<weights.size();j++){
           // weights.get(j).minusEquals(weightCostGradient.get(j).times(learningRate));
    @Override
    protected void update() {

        iterations++;
    }


    public int getIterationThreshold() {
        return iterationThreshold;
    }

    public void setIterationThreshold(int iterationThreshold) {
        this.iterationThreshold = iterationThreshold;
    }

    public int getGradientTheshhold() {
        return gradientTheshhold;
    }

    public void setGradientTheshhold(int gradientTheshhold) {
        this.gradientTheshhold = gradientTheshhold;
    }

}
