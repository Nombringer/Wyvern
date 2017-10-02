package Network.Trainer;

import Jama.Matrix;

import java.util.ArrayList;

/**
 * Created by fabd on 3/10/17.
 */
public interface UpdateRule {
    void update(ArrayList<Matrix> weights, ArrayList<Matrix> weightGradients);
}
