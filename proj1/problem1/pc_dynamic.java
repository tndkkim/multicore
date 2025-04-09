package problem1;

import java.util.concurrent.atomic.AtomicInteger;

public class pc_dynamic {
    private static int NUM_END = 200000;  // default input
    private static int NUM_THREADS = 6;   // default number of threads
    private static AtomicInteger nextStartIndex = new AtomicInteger(0);

    public static void main (String[] args) {
        if (args.length==2) {
            NUM_THREADS = Integer.parseInt(args[0]);
            NUM_END = Integer.parseInt(args[1]);
        }

        nextStartIndex.set(0);

        PrimeCounter[] counters = new PrimeCounter[NUM_THREADS];
        Thread[] threads = new Thread[NUM_THREADS];

        long startTime = System.currentTimeMillis();

        for (int i = 0; i < NUM_THREADS; i++) {
            counters[i] = new PrimeCounter(i);
            threads[i] = new Thread(counters[i]);
            threads[i].start();
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
        double performance = 1.0 / (timeDiff / 1000.0);  // performance = 1/execution_time

        System.out.println("Program Execution Time: " + timeDiff + "ms");
        System.out.println("Performance: " + performance + " operations per second");
        System.out.println("1..." + (NUM_END-1) + " prime# counter=" + totalPrimes);
    }

    static class PrimeCounter implements Runnable {
        private int threadId;
        private int counter;
        private long executionTime;

        public PrimeCounter(int threadId) {
            this.threadId = threadId;
            this.counter = 0;
        }

        @Override
        public void run() {
            long startTime = System.currentTimeMillis();

            while (true) {
                int start = nextStartIndex.getAndAdd(10);

                if (start >= NUM_END) {
                    break;
                }
                int end = Math.min(start + 10, NUM_END);
                for (int i = start; i < end; i++) {
                    if (isPrime(i)) {
                        counter++;
                    }
                }
            }

            long endTime = System.currentTimeMillis();
            executionTime = endTime - startTime;
        }

        public int getCounter() {
            return counter;
        }

        public long getExecutionTime() {
            return executionTime;
        }
    }

    private static boolean isPrime(int x) {
        int i;
        if (x <= 1) return false;
        for (i = 2; i < x; i++) {
            if (x % i == 0) return false;
        }
        return true;
    }
}