import Jama.Matrix;

/**
 * Created by fabd on 11/08/17.
 */
public abstract class ActivationFunction {

    static double momentumTerm = 0.01;

    abstract double apply(double input);

    abstract Matrix apply(Matrix input);

    abstract double applyGradFunc(double input);

    abstract Matrix applyGradFunc(Matrix input);

    public static double getMomentumTerm() {
        return momentumTerm;
    }

    public static void setMomentumTerm(double momentumTerm) {
        ActivationFunction.momentumTerm = momentumTerm;
    }
}
