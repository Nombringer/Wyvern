package Network.Trainer;

import Jama.Matrix;
import Network.NeuralNet;

import java.util.ArrayList;

/**
 * Created by fabd on 3/10/17.
 */
public class NaiveGradientDescent extends GradientDescentTrainer {

    private int iterations;
    private int iterationThreshold;
    private int gradientTheshhold;

    private double learningRate = 0.1;
    private double costThreshold = 0.0000001;
    private final int maxIterations = 100000;



    NaiveGradientDescent(NeuralNet net, Matrix tIn, Matrix tOut) {
        super(net, tIn, tOut);
    }

    UpdateRule updateRule = (ArrayList<Matrix> weights, ArrayList<Matrix> gradient) -> {
        for (int i = 0; i < weights.size(); i++) {
            weights.get(i).minusEquals(gradient.get(i).times(learningRate));
        }
    };


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
        trainingNet.applyUpdateRule();
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
