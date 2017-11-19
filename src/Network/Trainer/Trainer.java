package Network.Trainer;

import Jama.Matrix;

/**
 * Created by fabd on 28/09/17.
 */
public interface Trainer {
    void train();
    void setTrainingMatrices(Matrix trainingIn, Matrix trainingOut);
}
