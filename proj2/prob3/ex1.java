package prob3;

import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.BlockingQueue;

public class ex1 {
    public static void main(String[] args) {
        BlockingQueue<String> queue = new ArrayBlockingQueue<>(3);

        Thread producer = new Thread(new Producer(queue));
        Thread consumer = new Thread(new Consumer(queue));

        producer.start();
        consumer.start();
    }
}

class Producer implements Runnable {
    private BlockingQueue<String> queue;

    public Producer(BlockingQueue<String> queue) {
        this.queue = queue;
    }

    @Override
    public void run() {
        for (int i = 1; i <= 10; i++) {
            String message = i + "th message";
            System.out.println("Producer " + i);
            try {
                System.out.println("Trying to put: " + message);
                queue.put(message);
                System.out.println("Put " + message + " Done. Queue Size:" + queue.size());

                Thread.sleep((int)(Math.random() * 2000));
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        }
    }
}

class Consumer implements Runnable {
    private BlockingQueue<String> queue;

    public Consumer(BlockingQueue<String> queue) {
        this.queue = queue;
    }

    @Override
    public void run() {
        String message;

        for (int i = 1; i <= 10; i++) {
            System.out.println("Consumer " + i);
            try {
                System.out.println("Trying to take()");
                message = queue.take();
                System.out.println("take: " + message + " Done. Queue Size:" + queue.size());
                Thread.sleep((int)(Math.random() * 2000));
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        }
    }
}