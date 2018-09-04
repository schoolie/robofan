import time
import RPi.GPIO as GPIO



def move(revolutions, direction):
    
    GPIO.setmode(GPIO.BCM)

    pins = {
        'step': 18,
        'dir':  23,
    }

    for name, pin_num in pins.items():
        GPIO.setup(pin_num, GPIO.OUT)

    microstep_resolution = 1 
    steps_per_rev = 200
    step_delay = 0.004
    
    steps = int(revolutions * steps_per_rev * microstep_resolution)
    
    GPIO.output(pins['dir'], direction)

    for n in range(steps):
    
        GPIO.output(pins['step'], GPIO.HIGH)
        time.sleep(step_delay)
        GPIO.output(pins['step'], GPIO.LOW)
        time.sleep(step_delay)

    GPIO.cleanup() # cleanup all GPIO
    
if __name__ == '__main__':
    move(2, 0)
    time.sleep(0.25)
    move(2, -1)



