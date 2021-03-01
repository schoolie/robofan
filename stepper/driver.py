import time
import pigpio

default_pins = dict(
    enable=6,
    ms1=5,
    ms2=25,
    ms3=24,
    reset=23,
    sleep=22,
    step=17,
    direction=4,
)

microstep_truth_table = [  #(ms1, ms2, ms3)
    (0,0,0), # whole step
    (1,0,0), # half step
    (0,1,0), # quarter step
    (1,1,0), # eigth step
    (1,1,1), # sixteenth step
]

power_map = {
    1: 0,
    2: 1,
    4: 2,
    8: 3,
    16: 4,
}

class StepperDriver():
    def __init__(self, pins=default_pins):
        self.pins = pins
        self.pi = pigpio.pi()

        for name, pin_num in self.pins.items():
            self.pi.set_mode(pin_num, pigpio.OUTPUT)

        self.pi.write(self.pins['reset'], 1) 
        self.pi.write(self.pins['sleep'], 1) 


    def move(self, revolutions, direction, microstep_resolution=8, rev_per_sec=1):
        
        ## set microstep pins
        ms1, ms2, ms3 = microstep_truth_table[
            power_map.get(microstep_resolution, 3)
        ]
        self.pi.write(self.pins['ms1'], ms1) 
        self.pi.write(self.pins['ms2'], ms2) 
        self.pi.write(self.pins['ms3'], ms3) 

        print(ms1, ms2, ms3)

        rev_per_sec = float(rev_per_sec)
        steps_per_rev = 200
        
        step_delay = 1 / (steps_per_rev * microstep_resolution * rev_per_sec)
        
        steps = int(revolutions * steps_per_rev * microstep_resolution)
            
        ## Drive motor
        self.pi.write(self.pins['enable'], 0) ## enable FETs
        self.pi.write(self.pins['direction'], direction) ## Set direction pin
        
        wf=[]
        self.pi.wave_clear() # start a new waveform

        print(steps, step_delay*1e6)
        for n in range(steps):
            wf.append(pigpio.pulse(1<<self.pins['step'], 0, int(step_delay*1e6 / 2)))
            wf.append(pigpio.pulse(0, 1<<self.pins['step'], int(step_delay*1e6 / 2)))

            if n % 1000 == 0 and n > 0:
                print(n, len(wf))
                self.pi.wave_add_generic(wf)
                wf = []
                wf.append(pigpio.pulse(0, 0, self.pi.wave_get_micros()))
        
        self.pi.wave_add_generic(wf)

        wid = self.pi.wave_create()
        self.pi.wave_send_once(wid)

        print(self.pi.wave_get_micros() * 1e-6)

        move_time = self.pi.wave_get_micros() * 1e-6
        time.sleep(move_time + 0.1)
        self.pi.write(self.pins['enable'], 1) ## disable FETs
        

#     GPIO.cleanup() # cleanup all GPIO

if __name__ == '__main__':
    stepper_driver = StepperDriver()

    revs = 2.2
    stepper_driver.move(revs, 0, microstep_resolution=8, rev_per_sec = 1)
    time.sleep(0.2)
    stepper_driver.move(revs, 1, microstep_resolution=8, rev_per_sec = 1)
        
    # time.sleep(2)
    # stepper_driver.move(1, 1)
    # stepper_driver.move(1, 1)
    # stepper_driver.move(1, 1)
    # stepper_driver.move(1, 1)
