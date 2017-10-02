package Network.Trainer;


import Jama.Matrix;
import Network.NeuralNet;

import static Ultil.MatrixUtils.normaliseMatrix;

/**
 * Created by fabd on 28/09/17.
 */
public abstract class TrainerImpl implements Trainer {
    protected NeuralNet trainingNet;

    protected Matrix trainingIn;
    protected Matrix trainingOut;

    protected Matrix normalisedIn;
    protected Matrix normalisedOut;

    TrainerImpl(NeuralNet net, Matrix tIn, Matrix tOut) {
        this.trainingIn = tIn;
        this.trainingOut = tOut;
        trainingNet = net;
    }

    public abstract void Train(NeuralNet net, Matrix trainingInputs, Matrix trainingOutputs);

    private void NormaliseTrainingMatrices(){
        normalisedIn = normaliseMatrix(trainingIn);
        normalisedOut = normaliseMatrix(trainingOut);
    }
}
