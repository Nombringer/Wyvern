package Network.ActivationFunction;

import Jama.Matrix;

/**
 * Network.ActivationFunctionImpl.LogisticFunctionImpl activation function
 */
  public class LogisticFunctionImpl extends ActivationFunctionImpl {

    @Override
    public double apply(double z) {
        return (1/(1 + Math.exp(-z)));
    }

    @Override
    public Matrix apply(Matrix input) {
        Matrix out = new Matrix(input.getRowDimension(), input.getColumnDimension());

        for (int i = 0; i < input.getRowDimension(); i++) {
            for (int j = 0; j < input.getColumnDimension(); j++) {

                out.set(i, j, apply(input.get(i,j)));
            }
        }
        return out;
    }

    @Override
    public double applyGradFunc(double input) {
        return apply(input)*(1-apply(input));
    }

    @Override
    public Matrix applyGradFunc(Matrix input) {
        Matrix out = apply(input);

        for (int i = 0; i < input.getRowDimension(); i++) {
            for (int j = 0; j < input.getColumnDimension(); j++) {

                out.set(i, j, 1 - apply(input.get(i,j)));
            }
        }
       return out;
    }

    @Override
    public ActivationFunctionImpl copy() {
        return new LogisticFunctionImpl();
    }
}
