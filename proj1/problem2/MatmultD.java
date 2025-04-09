package problem2;

import java.util.*;
import java.lang.*;

// command-line execution example) java MatmultD 6 < mat500.txt
// 6 means the number of threads to use
// < mat500.txt means the file that contains two matrices is given as standard input
//
// In eclipse, set the argument value and file input by using the menu [Run]->[Run Configurations]->{[Arguments], [Common->Input File]}.

// Original JAVA source code: http://stackoverflow.com/questions/21547462/how-to-multiply-2-dimensional-arrays-matrix-multiplication
public class MatmultD {
    private static Scanner sc = new Scanner(System.in);

    public static void main(String[] args) {
        // block decomposition 이용
        //행을 블록 개수만큼 나눠서 각각 할당

        int threadCount = 1;
        if (args.length == 1) {
            threadCount = Integer.valueOf(args[0]);
        }

        int[][] a = readMatrix();
        int[][] b = readMatrix();

        long startTime = System.currentTimeMillis();

        MatrixMultiplier[] multipliers = new MatrixMultiplier[threadCount];
        Thread[] threads = new Thread[threadCount];
        int[][] c = new int[a.length][b[0].length]; // 결과 행렬

        int rowsPerThread = a.length / threadCount;
        int remainingRows = a.length % threadCount;

        int startRow = 0;
        for (int i = 0; i < threadCount; i++) {
            int numRows = rowsPerThread + (i < remainingRows ? 1 : 0);
            int endRow = startRow + numRows;

            multipliers[i] = new MatrixMultiplier(a, b, c, startRow, endRow, i);
            threads[i] = new Thread(multipliers[i]);
            threads[i].start();

            startRow = endRow;
        }

        for (int i = 0; i < threadCount; i++) {
            try {
                threads[i].join();
                System.out.println("Thread " + i + " execution time: " + multipliers[i].getExecutionTime() + "ms");
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        }

        long endTime = System.currentTimeMillis();
        long totalTimeDiff = endTime - startTime;
        double performance = 1.0 / (totalTimeDiff / 1000.0);

        System.out.printf("[thread_no]:%2d , [Time]:%4d ms\n", threadCount, totalTimeDiff);
        System.out.println("Performance: " + performance);

        printMatrix(c);
    }

    public static int[][] readMatrix() {
        int rows = sc.nextInt();
        int cols = sc.nextInt();
        int[][] result = new int[rows][cols];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result[i][j] = sc.nextInt();
            }
        }
        return result;
    }

    public static void printMatrix(int[][] mat) {
        //System.out.println("Matrix[" + mat.length + "][" + mat[0].length + "]");
        int rows = mat.length;
        int columns = mat[0].length;
        int sum = 0;
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < columns; j++) {
                //System.out.printf("%4d ", mat[i][j]);
                sum += mat[i][j];
            }
            //System.out.println();
        }
        System.out.println();
        System.out.println("Matrix Sum = " + sum + "\n");
    }

    static class MatrixMultiplier implements Runnable {
        private int[][] a;
        private int[][] b;
        private int[][] c;
        private int startRow;
        private int endRow;
        private int threadId;
        private long executionTime;

        public MatrixMultiplier(int[][] a, int[][] b, int[][] c, int startRow, int endRow, int threadId) {
            this.a = a;
            this.b = b;
            this.c = c;
            this.startRow = startRow;
            this.endRow = endRow;
            this.threadId = threadId;
        }

        @Override
        public void run() {
            long startTime = System.currentTimeMillis();

            int n = a[0].length;
            int p = b[0].length;

            for (int i = startRow; i < endRow; i++) {
                for (int j = 0; j < p; j++) {
                    c[i][j] = 0;
                    for (int k = 0; k < n; k++) {
                        c[i][j] += a[i][k] * b[k][j];
                    }
                }
            }

            long endTime = System.currentTimeMillis();
            executionTime = endTime - startTime;
        }

        public long getExecutionTime() {
            return executionTime;
        }
    }
}