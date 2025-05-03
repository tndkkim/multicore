package prob2;
import java.util.concurrent.Semaphore;

class ParkingSem {
    private Semaphore semaphore;

    public ParkingSem(int places) {
        this.semaphore = new Semaphore(places);
    }
    public void enter() { // enter parking garage
        try {
            semaphore.acquire();
            //System.out.println(" ");
        } catch (InterruptedException e) {}
    }
    public void leave(){ // leave parking garage
        semaphore.release();
        //System.out.println(" ");
    }
    public synchronized int getPlaces()
    {
        return semaphore.availablePermits();
    }
}

class Car extends Thread {
    private ParkingSem parkingSem;
    public Car(String name, ParkingSem p) {
        super(name);
        this.parkingSem = p;
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
            parkingSem.enter();
            justEntered();
            try {
                sleep((int)(Math.random() * 20000)); // stay within the parking garage
            } catch (InterruptedException e) {}
            aboutToLeave();
            parkingSem.leave();
            Left();

        }
    }
}

public class ParkingSemaphore {
    public static void main(String[] args){
        ParkingSem parkingQueue = new ParkingSem(7);
        for (int i=1; i<= 10; i++) {
            Car c = new Car("Car "+i, parkingQueue);
        }
    }
}