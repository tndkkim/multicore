package prob3;

import java.util.concurrent.BrokenBarrierException;
import java.util.concurrent.CyclicBarrier;

public class ex4 {
    public static void main(String[] args) {
        int num = 3;
        CyclicBarrier barrier = new CyclicBarrier(num, () ->
                System.out.println("Done : threads start moving to next phase"));
        /*
        Runnable barrierAction = new Runnable() {
            public void run() {
                System.out.println("Done : threads start moving to next phase");
            }
        };
        CyclicBarrier barrier = new CyclicBarrier(2, barrierAction);

        */

        for (int i = 0; i < num; i++) {
            Thread thread = new Thread(new thread(i+1, barrier));
            thread.start();
        }
    }
}

class thread implements Runnable {
    private int id;
    private CyclicBarrier barrier;

    public thread(int id, CyclicBarrier barrier) {
        this.id = id;
        this.barrier = barrier;
    }

    @Override
    public void run() {
        try {
            System.out.println("thread " + id + " starts first cycle");
            Thread.sleep(500 * id);
            System.out.println("thread " + id + " finishes first cycle");

            System.out.println("thread " + id + " waits for others at first barrier");
            barrier.await();

            System.out.println("thread " + id + " starts second cycle");
            Thread.sleep(300 * id);
            System.out.println("thread " + id + " finishes second cycle");

            System.out.println("thread " + id + " waits for others at second barrier");
            barrier.await();

            System.out.println("thread " + id + " completed all work");

        } catch (InterruptedException | BrokenBarrierException e) {
            e.printStackTrace();
        }
    }
}