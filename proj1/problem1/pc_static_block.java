package problem1;

public class pc_static_block {
    private static int NUM_END = 200000;  // default input
    private static int NUM_THREADS = 6;   // default number of threads

    public static void main (String[] args) {
        if (args.length==2) {
            NUM_THREADS = Integer.parseInt(args[0]);
            NUM_END = Integer.parseInt(args[1]);
        }

        totalCounter[] counters = new totalCounter[NUM_THREADS];
        Thread[] threads = new Thread[NUM_THREADS];

        long startTime = System.currentTimeMillis();

        for (int i = 0; i < NUM_THREADS; i++) {
            counters[i] = new totalCounter(i);
            threads[i] = new Thread(counters[i]);
            threads[i].start(); //counters[i].run() 실행
        }

        int totalPrimes = 0;
        for (int i = 0; i < NUM_THREADS; i++) {
            try {
                threads[i].join();
                totalPrimes += counters[i].getCounter();
                System.out.println("Thread " + i + " execution time: " + counters[i].getExecutionTime() + "ms");
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        }

        long endTime = System.currentTimeMillis();
        long timeDiff = endTime - startTime;
        double performance = 1.0 / (timeDiff / 1000.0);

        System.out.println("Program Execution Time: " + timeDiff + "ms");
        System.out.println("Performance: " + performance + " operations per second");
        System.out.println("1..." + (NUM_END-1) + " prime# counter=" + totalPrimes);
    }

    private static class totalCounter implements Runnable {
        private int threadId;
        private int counter;
        private long timeDiff;

        public totalCounter(int threadId) {
            this.threadId = threadId;
            this.counter = 0;
        }

        @Override
        public void run() {
            int blockSize = NUM_END / NUM_THREADS;
            int start = threadId * blockSize;
            int end = (threadId == NUM_THREADS - 1) ? NUM_END : (threadId + 1) * blockSize-1;
            long startTime = System.currentTimeMillis();

            for (int i = start; i <= end; i++) {
                if (isPrime(i)) {
                    counter++;
                }
            }
            long endTime = System.currentTimeMillis();
            timeDiff = endTime - startTime;
        }

        public int getCounter() {
            return counter;
        }

        public long getExecutionTime() {
            return timeDiff;
        }
    }

    private static boolean isPrime(int x) {
        int i;
        if (x<=1) return false;
        for (i=2;i<x;i++) {
            if (x%i == 0)  return false;
        }
        return true;
    }
}