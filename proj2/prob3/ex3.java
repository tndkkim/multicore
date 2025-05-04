package prob3;

import java.util.concurrent.atomic.AtomicInteger;

public class ex3 {
    public static void main(String[] args) throws InterruptedException {
        Counter counter = new Counter();

        Thread t1 = new Thread(counter, "t1");
        Thread t2 = new Thread(counter, "t2");

        t1.start();
        t2.start();

        t1.join();
        t2.join();

        // Display the final counter value
        System.out.println("Final : " + counter.getValue());
    }
}

class Counter implements Runnable{
    private AtomicInteger value = new AtomicInteger(0);

    @Override
    public void run() {
        String threadName = Thread.currentThread().getName();

        if (threadName.equals("t1")) {
            setValue(10);
            getAndIncrement(5);
            incrementAndGet(3);
        } else {
            setValue(20);
            getAndIncrement(7);
            incrementAndGet(2);
        }

        System.out.println(threadName + " finished. Current value: " + getValue());
    }

    private int incrementAndGet(int i) {
        int newVal = value.addAndGet(i);
        System.out.println(Thread.currentThread().getName() + " addAndGet('" + i + "') return : " + newVal + ", current val : " + value.get());
        return newVal;
    }

    private int getAndIncrement(int i) {
        int prevValue = value.getAndAdd(i);
        System.out.println(Thread.currentThread().getName() + " getAndAdd('" + i + "') return : " + prevValue + ", current val : " + value.get());
        return prevValue;


    }

    public int getValue() {
        return value.get();
    }

    public void setValue(int newValue){
        value.set(newValue);
        System.out.println(Thread.currentThread().getName() + " set to " + newValue);
    }


}