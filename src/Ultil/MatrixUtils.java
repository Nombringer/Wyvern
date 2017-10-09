package Ultil;

import Jama.Matrix;

/**
 * Created by fabd on 28/09/17.
 */
public class MatrixUtils {

    /**
     * @param matrix
     * @return The matrix, with the data normalised to 1 based on the maximum value
     */
    public static Matrix normaliseMatrix(Matrix matrix){
        Matrix normalised;
        normalised = matrix.times(1/getMax(matrix));
        return normalised;
    }



    /**
     *
     * @param matrix
     * @return The maximum value in the matrix
     */
    public static double getMax(Matrix matrix) {
        double max = Double.MIN_VALUE;

        for (int i = 0; i < matrix.getRowDimension(); i++) {
            for (int j = 0; j < matrix.getColumnDimension(); j++) {

                max = Math.max(max, matrix.get(i, j));
            }

        }
        return max;
    }

    /**
     * Computes the sum the elements of a matrix.
     *
     * @param m the matrix.
     * @return the sum of the elements of the matrix
     */
    public static double sum(Matrix m) {
        int numRows = m.getRowDimension();
        int numCols = m.getColumnDimension();
        double sum = 0;
        // loop through the rows and compute the sum
        for (int i = 0; i < numRows; i++) {
            for (int j = 0; j < numCols; j++) {
                sum += m.get(i, j);
            }
        }
        return sum;
    }

    /**
     * Gets the specified row of a matrix.
     *
     * @param m   the matrix.
     * @param row the row to get.
     * @return the specified row of m.
     */
    public static Matrix getrow(Matrix m, int row) {
        return m.getMatrix(row, row, 0, m.getColumnDimension() - 1);
    }

    /**
     * Gets the specified column of a matrix.
     *
     * @param m   the matrix.
     * @param col the column to get.
     * @return the specified column of m.
     */
    public static Matrix getcol(Matrix m, int col) {
        return m.getMatrix(0, m.getRowDimension() - 1, col, col);
    }




}
