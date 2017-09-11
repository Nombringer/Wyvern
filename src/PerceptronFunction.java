import Jama.Matrix;

/**
 * Created by fabd on 12/09/17.
 */
public class PerceptronFunction extends ActivationFunction {

    @Override
    double apply(double input) {
        if(input<0){return -1;}
        return 1;
    }

    @Override
    Matrix apply(Matrix input) {

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
    double applyGradFunc(double input) {
        return 0;
    }

    @Override
    Matrix applyGradFunc(Matrix input) {
        return null;
    }
}
