package Network.ActivationFunction;

import Jama.Matrix;

/**
 * Created by fabd on 5/09/17.
 */
public class HyperbolicTangent extends ActivationFunctionImpl {


    @Override
    public double apply(double input) {
       double x = input*(2d/3d);
       x = 1.7159*Math.tanh(x) + momentumTerm*x;
       return x;
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

        return ((1- (Math.pow(Math.exp(input) - Math.exp(-input), 2))/
                    Math.pow(Math.exp(input) + Math.exp(-input), 2)));
    }

    @Override
    public Matrix applyGradFunc(Matrix input) {
        Matrix out = apply(input);

        for (int i = 0; i < input.getRowDimension(); i++) {
            for (int j = 0; j < input.getColumnDimension(); j++) {

                out.set(i, j, applyGradFunc(input.get(i,j)));
            }
        }
        return out;
    }

    @Override
    public ActivationFunctionImpl copy() {
        return new HyperbolicTangent();
    }
}
