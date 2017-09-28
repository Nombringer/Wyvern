package Network.ActivationFunction;

import Jama.Matrix;

/**
 * Created by fabd on 28/09/17.
 */
public interface ActivationFunction {

    abstract double apply(double input);

    abstract Matrix apply(Matrix input);

    abstract double applyGradFunc(double input);

    abstract Matrix applyGradFunc(Matrix input);

    abstract ActivationFunction copy();

}
