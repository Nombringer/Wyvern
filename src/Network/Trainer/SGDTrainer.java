package Network.Trainer;

import Jama.Matrix;
import Network.NeuralNet;

import java.util.ArrayList;
import java.util.concurrent.ThreadLocalRandom;

import static Util.MatrixUtils.getrow;
import static Util.MatrixUtils.setrow;


/**
 * Created by fabd on 7/10/17.
 */
public class SGDTrainer extends GradientDescentTrainer {

    private double learningRate = 0.01; //TODO: Set properly

    private int epochs;
    private int epochLimit = 10000; //TODO: Something better
    private int miniBatchSize;
    private double costThreshold = 0.0000001;


    Matrix fullIn;
    Matrix fullOut;

    public SGDTrainer(NeuralNet net, Matrix tIn, Matrix tOut, int batchSize) {
        super(net, tIn, tOut);
        fullIn = trainingIn;
        fullOut = trainingOut;
        miniBatchSize = batchSize;
        setMiniBatch();
    }

    @Override
    public void train() {
        epochs = 0;
        setMiniBatch();
        super.train();

    }

    private void setMiniBatch(){
        Matrix miniBatchIn = new Matrix(miniBatchSize, fullIn.getColumnDimension());
        Matrix miniBatchOut = new Matrix(miniBatchSize, fullOut.getColumnDimension());

        for (int i = 0; i < miniBatchSize -1; i++){
            int random = ThreadLocalRandom.current().nextInt(0, fullIn.getRowDimension());
            setrow(miniBatchIn, i , getrow(fullIn, random));
            setrow(miniBatchOut, i, getrow(fullOut, random));
        }
        trainingIn = miniBatchIn;
        trainingOut = miniBatchOut;
    }

    @Override
    protected boolean terminationCondition() {
        if (epochs%100 == 1){System.out.println(epochs); trainingNet.computeCost(trainingNet.computeEstimates(trainingIn));}

        if (epochs==0){return false;}
        if (epochs>=epochLimit||trainingNet.getCost()<costThreshold){return true;}
        return false;
    }



    @Override
    protected void update() {
        trainingNet.applyUpdateRule(updateRule);
        epochs++;
    }


    UpdateRule updateRule = (ArrayList<Matrix> weights, ArrayList<Matrix> gradient) -> {
        for (int i = 0; i < weights.size(); i++) {
            weights.get(i).minusEquals(gradient.get(i).times(learningRate));
        }
    };



    public int getMiniBatchSize() {
        return miniBatchSize;
    }


    public void setMiniBatchSize(int miniBatchSize) {
        if (miniBatchSize>fullIn.getColumnDimension()) {throw new IllegalArgumentException("Minibatch greater than output size");}
        this.miniBatchSize = miniBatchSize;
    }


}


