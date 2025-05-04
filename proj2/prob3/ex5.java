package prob3;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.*;

public class ex5{
    private static final int NUM_END = 200000;  // default input
    private static final int NUM_THREADS = 6;

    public static void main(String[] args){
        ExecutorService executor = Executors.newFixedThreadPool(NUM_THREADS);
        List<Future<Integer>> futures = new ArrayList<>();

        for (int i = 0; i < NUM_THREADS; i++){
            int start = i*(NUM_END/NUM_THREADS) + 1;
            int end;
            if (i == NUM_THREADS - 1) {
                end = NUM_END;
            } else {
                end = (i + 1) * (NUM_END / NUM_THREADS);
            }

            PrimeCounter counter = new PrimeCounter(start, end);
            Future<Integer> future = executor.submit(counter);
            futures.add(future);

        }

        int totalPrimes = 0;
        try{
            for (Future<Integer> future : futures){
                totalPrimes += future.get();
            }
        } catch (InterruptedException | ExecutionException e){e.printStackTrace();}

        executor.shutdown();
        System.out.println("answer is " + totalPrimes);
    }
}

class PrimeCounter implements Callable<Integer>{
    private final int start;
    private final int end;

    public PrimeCounter(int start, int end) {
        this.start = start;
        this.end = end;
    }
    @Override
    public Integer call() {
        int count = 0;
        for (int i = start; i <= end; i++) {
            if (isPrime(i)) {
                count++;
            }
        }
        System.out.println("Range [" + start + ", " + end + "]: found " + count + " primes");
        return count;
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