package Network.Trainer;

import Jama.Matrix;
import Network.NeuralNet;

/**
 * Created by fabd on 28/09/17.
 */
public interface Trainer {

    void Train(NeuralNet net, Matrix trainingInputs, Matrix trainingOutputs);


}
