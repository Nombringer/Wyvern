package Network.ActivationFunction;

/**
 * Created by fabd on 11/08/17.
 */
public abstract class ActivationFunctionImpl implements ActivationFunction {

    //TODO This should be rewritten with lambda's somehow

    static double momentumTerm = 0.01;

    public static double getMomentumTerm() {
        return momentumTerm;
    }

    public static void setMomentumTerm(double momentumTerm) {
        ActivationFunctionImpl.momentumTerm = momentumTerm;
    }

}
