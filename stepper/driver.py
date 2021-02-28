import time
import pigpio

class StepperDriver():
    def __init__(self, pins=dict(step=18, direction=23, enable=24)):
        self.pins = pins
        self.pi = pigpio.pi()

        for name, pin_num in self.pins.items():
            self.pi.set_mode(pin_num, pigpio.OUTPUT)

    def move(self, revolutions, direction, rev_per_sec=1):

        rev_per_sec = float(rev_per_sec)
        microstep_resolution = 8
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
            wf.append(pigpio.pulse(1<<self.pins['step'], 0, 1 * step_delay*1e6))
            wf.append(pigpio.pulse(0, 1<<self.pins['step'], 1 * step_delay*1e6))
        
        self.pi.wave_add_generic(wf)

        wid = self.pi.wave_create()
        self.pi.wave_send_once(wid)

        time.sleep(0.1)
        self.pi.write(self.pins['enable'], 1) ## disable FETs
        

#     GPIO.cleanup() # cleanup all GPIO

if __name__ == '__main__':
    stepper_driver = StepperDriver()

    stepper_driver.move(0.5, 0)
    time.sleep(2)
    stepper_driver.move(0.5, 1)
