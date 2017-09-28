package Network.ActivationFunction;

import Jama.Matrix;

/**
 * Created by fabd on 12/09/17.
 */
public class PerceptronFunctionImpl extends ActivationFunctionImpl {

    @Override
    public double apply(double input) {
        if(input<0){return -1;}
        return 1;
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

    //TODO: Figure out if these have any meaning
    @Override
    public double applyGradFunc(double input) {
        return 0;
    }

    @Override
    public Matrix applyGradFunc(Matrix input) {
        return null;
    }

    @Override
    public ActivationFunctionImpl copy() {
        return new PerceptronFunctionImpl();
    }
}
