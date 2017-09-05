import Jama.Matrix;

/**
 * Created by fabd on 11/08/17.
 */
public interface ActivationFunction {

    double apply(double input);

    Matrix apply(Matrix input);

    double applyGradFunc(double input);

    Matrix applyGradFunc(Matrix input);

}
