package prob1;

import java.util.concurrent.*;

class ParkingQueue {
    private BlockingQueue<String> queue;

    public ParkingQueue(int places) {
        this.queue = new ArrayBlockingQueue<>(places);
        for (int i = 0; i < places; i++) {
            try {
                queue.put("1");
            } catch (InterruptedException e) {}
        }
    }
    public void enter() { // enter parking garage
        try {
            queue.take();
            //System.out.println("Queue size: " + queue.size() + " available");
        } catch (InterruptedException e) {}
    }
    public void leave(){ // leave parking garage
        try{
            queue.put("1");
            //System.out.println("Queue size: " + queue.size() + " available");
        } catch (InterruptedException e){}
    }
    public int getPlaces()
    {
        return queue.size();
    }
}



class Car extends Thread {
    private ParkingQueue parkingQueue;
    public Car(String name, ParkingQueue p) {
        super(name);
        this.parkingQueue = p;
        start();
    }

    private void tryingEnter()
    {
        System.out.println(getName()+": trying to enter");
    }


    private void justEntered()
    {
        System.out.println(getName()+": just entered");

    }

    private void aboutToLeave()
    {
        System.out.println(getName()+":                                     about to leave");
    }

    private void Left()
    {
        System.out.println(getName()+":                                     have been left");
    }

    public void run() {
        while (true) {
            try {
                sleep((int)(Math.random() * 10000)); // drive before parking
            } catch (InterruptedException e) {}
            tryingEnter();
            parkingQueue.enter();
            justEntered();
            try {
                sleep((int)(Math.random() * 20000)); // stay within the parking garage
            } catch (InterruptedException e) {}
            aboutToLeave();
            parkingQueue.leave();
            Left();

        }
    }
}

public class ParkingBlockingQueue {
    public static void main(String[] args){
        ParkingQueue parkingQueue = new ParkingQueue(7);
        for (int i=1; i<= 10; i++) {
            Car c = new Car("Car "+i, parkingQueue);
        }
    }
}
