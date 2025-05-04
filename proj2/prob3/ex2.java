package prob3;

import java.util.concurrent.locks.ReadWriteLock;
import java.util.concurrent.locks.ReentrantReadWriteLock;

public class ex2 {
    public static void main(String[] args) {
        SharedData data = new SharedData();

        for (int i = 0; i < 5; i++) { //10 reader threads
            Thread reader = new Thread(new Reader(data, i));
            reader.start();
        }

        for (int i = 0; i < 3; i++) {
            Thread writer = new Thread(new Writer(data, i));
            writer.start();
        }
    }
}

class SharedData {
    private int data = 0;
    private ReadWriteLock lock = new ReentrantReadWriteLock();

    public int readData() { //multiple threads can lock the lock for reading
        lock.readLock().lock();
        try {
            System.out.println(Thread.currentThread().getName() + " start reading : " + data);
            Thread.sleep(1000);
            System.out.println(Thread.currentThread().getName() + " finish reading : " + data);
            return data;
        } catch (InterruptedException e) {
            e.printStackTrace();
            return -1;
        } finally {
            lock.readLock().unlock();
        }
    }

    public void writeData(int newValue) { //only one thread at a time can lock the lock for writing.
        lock.writeLock().lock();
        try {
            System.out.println(Thread.currentThread().getName() + " start writing ");
            Thread.sleep(2000);
            this.data = newValue;
            System.out.println(Thread.currentThread().getName() + " finish writing : " + data);
        } catch (InterruptedException e) {
            e.printStackTrace();
        } finally {
            lock.writeLock().unlock();
        }
    }
}

class Reader implements Runnable {
    private SharedData data;
    private int id;

    public Reader(SharedData data, int id) {
        this.data = data;
        this.id = id;
    }

    @Override
    public void run() {
        for (int i = 0; i < 3; i++) {
            int value = data.readData();
            System.out.println("Reader " + id + " value : " + value);
            try {
                Thread.sleep((int)(Math.random() * 1000));
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        }
    }
}

class Writer implements Runnable {
    private SharedData data;
    private int id;

    public Writer(SharedData data, int id) {
        this.data = data;
        this.id = id;
    }

    @Override
    public void run() {
        for (int i = 0; i < 2; i++) {
            int newValue = (int)(Math.random() * 100);
            data.writeData(newValue);
            System.out.println("Writer " + id + " edited value : " + newValue);
            try {
                Thread.sleep((int)(Math.random() * 2000));
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        }
    }
}