package Network.Trainer;

import Jama.Matrix;
import Network.NeuralNet;

import java.util.ArrayList;

/**
 * Created by fabd on 3/10/17.
 */
public class NaiveGradientDescent extends GradientDescentTrainer {

    private int iterations;
    private double learningRate = 0.1;
    private double costThreshold = 0.0000001;
    private int iterationThreshhold = 100000;


    /**
     *
     * @param net The NN to be trained
     * @param tIn The inputs of the training set
     * @param tOut The outputs of the Training Set
     */
    public NaiveGradientDescent(NeuralNet net, Matrix tIn, Matrix tOut) {
        super(net, tIn, tOut);
        iterations = 0;
    }



    @Override
    protected boolean terminationCondition() {
        if (iterations==0){return false;} //TODO: Ugly as hell avoiding of null pointers
        if (iterations%1000 == 1){System.out.println(iterations); trainingNet.printCurrentCost();}
        if(iterations>=iterationThreshhold||trainingNet.getCost()<costThreshold){
            iterations = 0;
            System.out.println("Training Completed");
            return true;
        }
        return false;
    }


    UpdateRule updateRule = (ArrayList<Matrix> weights, ArrayList<Matrix> gradient) -> {
        for (int i = 0; i < weights.size(); i++) {
            weights.get(i).minusEquals(gradient.get(i).times(learningRate));
        }
    };


    @Override
    protected void update() {
            trainingNet.applyUpdateRule(updateRule);
            iterations++;
    }

    public int getIterationThreshold() {return iterationThreshhold; }

    public void setIterationThreshold(int iterationThreshold) {
        this.iterationThreshhold = iterationThreshold;
    }


}
