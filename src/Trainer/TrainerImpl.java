package Trainer;


import Jama.Matrix;
import Network.NeuralNet;

/**
 * Created by fabd on 28/09/17.
 */
public abstract class TrainerImpl implements Trainer {

    private Matrix trainingIn;
    private Matrix trainingOut;

    TrainerImpl(NeuralNet net, Matrix tIn, Matrix tOut){
        this.trainingIn = tIn;
        this.trainingOut = tOut;
    }





}
