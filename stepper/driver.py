import time
import pigpio
import numpy as np

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

    def calc_step_durations(self, S_tgt, V_max, J, microsteps_per_rev):
    
        T = np.sqrt(4*V_max/J) ## Duration of Accel S Curve to reach V_max
        S = V_max/2 * T        ## Position change during Accel S Curve

        if S*2 > S_tgt:        ## If 2 S Curves (Accel, Decel) don't fit in total move request, calculate shortened S Curves
            S = S_tgt/2
            T = (8*S/J)**(1/3)
            
        V_hold = S/T * 2         ## Calculate Velocity at end of Accel S Curve (==V_max unless shortened)
        S_hold = S_tgt - S*2     ## Position change during constant velocity move
        T_hold = S_hold / V_hold ## Time of constant velocity move

        ## Define 5 zones of operation, time duration, and Jerk
        zones = [
            [T/2, J, None],
            [T/2, -J, None],
            [T_hold, 0, V_hold],
            [T/2, -J, None],
            [T/2, J, None],
        ]

        # Initialize calc variables
        t0 = 0
        s0 = 0
        v0 = 0
        a0 = 0
        last_microstep_s = 0
        last_pulse_time = 0
        results = []
        step_durations = []


        for duration, jerk, V_const in zones:
            
            if duration > 0:
                ## Constant velocity region is longer, so sensitive to numerical errors
                ## Since pulse duration is easily calculated, do that to eliminate errors
                if V_const is not None:  
                    dt = 1 / (V_const * microsteps_per_rev)
                    
                    for t in np.arange(0, duration, dt):
                        s = s0 + V_const * t
                        v = V_const
                        a = 0
                        
                        
                        step_durations.append(dt)
                        pulse = 1
                        results.append([t+t0, s, v, a, pulse])
                        
                        last_pulse_time = t + t0
                    t = t+dt

                ## If we're in the acceleration phase, use numerical method
                else:
                
                    for t in np.linspace(0, duration, 2000):
                        
                        ## Calculate position, velocity, and acceleration based on conditions at 
                        ## beginning of zone, and jerk input
                        s = s0 + v0*t + a0/2*t**2 + jerk/6*t**3
                        v = v0 + a0*t + jerk/2*t**2
                        a = a0 + jerk * t
                        
                        ## Slice position into microsteps
                        microstep_s = s * microsteps_per_rev       
                        pulse = 0

                        if microstep_s - last_microstep_s > 1:      
                            pulse = 1
                            step_durations.append(t + t0 - last_pulse_time) ## store duration of pulses 

                            last_microstep_s = microstep_s
                            last_pulse_time = t + t0

                        results.append([t+t0, s, v, a, pulse]) ## This list isn't used in operation, just for debug
                
                    
                t0 = t+t0
                s0 = s
                v0 = v
                a0 = a

        return step_durations[1:-1]


    def move(self, revolutions, direction, microstep_resolution=8, rev_per_sec=1, max_jerk=200):
        
        ## set microstep pins
        ms1, ms2, ms3 = microstep_truth_table[
            power_map.get(microstep_resolution, 3)
        ]
        self.pi.write(self.pins['ms1'], ms1) 
        self.pi.write(self.pins['ms2'], ms2) 
        self.pi.write(self.pins['ms3'], ms3) 

        rev_per_sec = float(rev_per_sec)
        steps_per_rev = 200
        microsteps_per_rev = steps_per_rev * microstep_resolution

        step_durations = self.calc_step_durations(revolutions, rev_per_sec, max_jerk, microsteps_per_rev)

        ## Drive motor
        self.pi.write(self.pins['enable'], 0) ## enable FETs
        self.pi.write(self.pins['direction'], direction) ## Set direction pin
        
        wf=[]
        self.pi.wave_clear() # start a new waveform

        # for n in range(steps):
        for n, duration in enumerate(step_durations):
            wf.append(pigpio.pulse(1<<self.pins['step'], 0, int(duration*1e6 / 2)))
            wf.append(pigpio.pulse(0, 1<<self.pins['step'], int(duration*1e6 / 2)))

            if n % 1000 == 0 and n > 0:
                self.pi.wave_add_generic(wf)
                wf = []
                wf.append(pigpio.pulse(0, 0, self.pi.wave_get_micros()))
        
        self.pi.wave_add_generic(wf)

        wid = self.pi.wave_create()
        self.pi.wave_send_once(wid)

        move_time = self.pi.wave_get_micros() * 1e-6
        time.sleep(move_time + 0.1)
        self.pi.write(self.pins['enable'], 1) ## disable FETs

        # import matplotlib.pyplot as plt
        # plt.figure(figsize=[12,8])
        # plt.plot(step_durations)
        # plt.yscale('log')
        # plt.show()

        return step_durations
        


if __name__ == '__main__':
    stepper_driver = StepperDriver()

    revs = 2.2
    stepper_driver.move(revs, 0, microstep_resolution=8, rev_per_sec = 1)
    time.sleep(0.2)
    stepper_driver.move(revs, 1, microstep_resolution=8, rev_per_sec = 1)
