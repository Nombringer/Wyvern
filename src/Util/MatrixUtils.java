package Util;

import Jama.Matrix;

/**
 * Created by fabd on 28/09/17.
 */
public class MatrixUtils {

    /**
     * Appends additional rows to the first matrix.
     *
     * @param m the first matrix.
     * @param n the matrix to append containing additional rows.
     * @return a matrix with all the rows of m then all the rows of n.
     */
    public static Matrix rowAppend(Matrix m, Matrix n) {
        int mNumRows = m.getRowDimension();
        int mNumCols = m.getColumnDimension();
        int nNumRows = n.getRowDimension();
        int nNumCols = n.getColumnDimension();

        if (mNumCols != nNumCols)
            throw new IllegalArgumentException("Number of columns must be identical to row-append.");

        Matrix x = new Matrix(mNumRows + nNumRows, mNumCols);
        x.setMatrix(0, mNumRows - 1, 0, mNumCols - 1, m);
        x.setMatrix(mNumRows, mNumRows + nNumRows - 1, 0, mNumCols - 1, n);

        return x;
    }

    /**
     * Appends additional columns to the first matrix.
     *
     * @param m the first matrix.
     * @param n the matrix to append containing additional columns.
     * @return a matrix with all the columns of m then all the columns of n.
     */
    public static Matrix columnAppend(Matrix m, Matrix n) {
        int mNumRows = m.getRowDimension();
        int mNumCols = m.getColumnDimension();
        int nNumRows = n.getRowDimension();
        int nNumCols = n.getColumnDimension();

        if (mNumRows != nNumRows)
            throw new IllegalArgumentException("Number of rows must be identical to column-append.");

        Matrix x = new Matrix(mNumRows, mNumCols + nNumCols);
        x.setMatrix(0, mNumRows - 1, 0, mNumCols - 1, m);
        x.setMatrix(0, mNumRows - 1, mNumCols, mNumCols + nNumCols - 1, n);

        return x;
    }


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
     * Sets the specified row of a matrix.  Modifies the passed matrix.
     *
     * @param m      the matrix.
     * @param row    the row to modify.
     * @param values the new values of the row.
     */
    public static void setrow(Matrix m, int row, Matrix values) {
        if (!isRowVector(values))
            throw new IllegalArgumentException("values must be a row vector.");
        m.setMatrix(row, row, 0, m.getColumnDimension() - 1, values);
    }

    /**
     * Sets the specified column of a matrix.  Modifies the passed matrix.
     *
     * @param m      the matrix.
     * @param col    the column to modify.
     * @param values the new values of the column.
     */
    public static void setcol(Matrix m, int col, Matrix values) {
        if (!isColumnVector(values))
            throw new IllegalArgumentException("values must be a column vector.");
        m.setMatrix(0, m.getRowDimension() - 1, col, col, values);
    }

    /**
     * Determines if a given matrix is a row vector, that is, it has only one row.
     *
     * @param m the matrix.
     * @return whether the given matrix is a row vector (whether it has only one row).
     */
    public static boolean isRowVector(Matrix m) {
        return m.getRowDimension() == 1;
    }

    /**
     * Determines if a given matrix is a column vector, that is, it has only one column.
     *
     * @param m the matrix.
     * @return whether the given matrix is a column vector (whether it has only one column).
     */
    public static boolean isColumnVector(Matrix m) {
        return m.getColumnDimension() == 1;
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
