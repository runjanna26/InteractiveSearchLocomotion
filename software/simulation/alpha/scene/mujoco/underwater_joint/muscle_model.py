"""
X. Xiong and P. Manoonpong, "Adaptive Motor Control for Human-like Spatial-temporal Adaptation," 
2018 IEEE International Conference on Robotics and Biomimetics (ROBIO), Kuala Lumpur, Malaysia, 2018, pp. 2107-2112, doi: 10.1109/ROBIO.2018.8665222.
"""

import numpy as np
import time

'''
=========[Experiment]=========
if b = 0.0
    a = 35.0
    very compliance

    a = 0.5
    more stiff

    a = 0.2
    very stiff *(can try to converge to zero error)

    a = 0.01
    unstable

if a = 1.0
    b = 1.0
    very compliance

    b = 5.0
    compliance

    b = 20.0
    stiff but can deviate

    b = 50.0
    more stiff but can deviate **(more sensitive with tracking error) 

    b = 100.0
    more stiff but can deviate **(more sensitive with tracking error) 

    b = 500.0
    more stiff but can deviate **(more sensitive with tracking error) 

    b = 1000.0
    more stiff but can deviate **(more sensitive with tracking error) 

    b = 2000.0
    unstable
    **  it mean that when we try to deviate the joint angle motor will produce the suddenly
        (but cannot convergence to zero error position)

[Definition]
self.a      : overall value of Adaptation Scalar
self.b      : sensitivity of Adaptation Scalar associate with tracking error
self.beta   : weight of tracking error associate with velocity tracking


[Combination 1]
a = 0.1
b = 50.0
very stiff and fast adaptation

[Combination 2]
a = 35.0
b = 1.0
very compliance and slow adaptation (deviate +- 120 degree)

[Combination 3]
a = 0.2
b = 5.0
optimal compliance and adaptation (deviate +- 30 degree)
'''
#20.0 1.0 0.5  max. tau 1.0 compliance

class MuscleModel:
    '''
    Muscle model for one limb
    '''
    def __init__(self, _a:float, _b:float, _beta:float, _init_pos, number_motor):
        self.pos_init           = _init_pos
        
        self.a                  = _a
        self.b                  = _b
        self.β                  = _beta
        self.γ                  = 0.0

        self.pos_des            = 0.0        
        self.pos_fb             = 0.0
        self.pos_des_prev       = 0.0 

        self.vel_des            = 0.0
        self.vel_fb             = 0.0

        self.K                  = 0.0
        self.D                  = 0.0
        self.F                  = 0.0
        self.tau                = 0.0
        

        self.timestamp_now      = 0.0 
        self.timestamp_prev     = 0.0
        self.timestamp_now = time.time() 

        self.transient_flag = True

    def calculate(self, pos_des, pos_fb, vel_fb):
        self.pos_des = pos_des
        self.pos_fb  = pos_fb
        self.vel_fb  = vel_fb

        self.timestamp_now = time.time() 
        Ts = (self.timestamp_now - self.timestamp_prev) 
        if Ts <= 0 or Ts > 0.5:
            Ts = 1e-3
        self.vel_des = (self.pos_des - self.pos_des_prev) / Ts




        # Torque oscillate
        self.F = self.gen_track_error() / self.gen_adapt_scalar()
        self.K = self.F * self.gen_pos_error()
        self.D = self.F * self.gen_vel_error()

        # Torque smooth
        # self.K = np.diag([1])
        # self.D = np.diag([0.1])
        

        # self.tau = - self.F - self.K * self.gen_pos_error() - self.D * self.gen_vel_error()



        self.timestamp_prev = self.timestamp_now
        self.pos_des_prev   = self.pos_des


    def gen_pos_error(self):
        '''
        Calculate position error (e)
        e = (q - q_init) - qd
        '''
        return ((self.pos_fb - self.pos_init) - self.pos_des)

    def gen_vel_error(self):
        '''
        Calculate velocity error (e_dot)
        e_dot = q_dot - qd_dot
        '''
        return (self.vel_fb - self.vel_des)

    def gen_track_error(self):
        '''
        Calculate tracking error
        '''
        return (self.gen_pos_error() + self.β * self.gen_vel_error())

    def gen_adapt_scalar(self):
        return self.a / (1.0 + (self.b * (np.abs(self.gen_track_error()))**2))
    
    def get_stiffness(self):
        return np.float64(self.limit_value(self.K, 0.0, 500.0))
    def get_damping(self):
        return np.float64(self.limit_value(self.D, 0.0, 5.0))
    def get_feedforward_force(self):
        return -np.float64(self.limit_value(self.F, -15.0, 15.0))
    
    def limit_value(self, value, min, max):
        """
        Limits value to be between min and max

        Args:
            value: The value to be limited.
            min: The lowest number allowed (inclusive) for value
            max: The highest number allowed (inclusive) for value
        """
        if value >= max:
            return max
        elif value <= min:
            return min
        else:
            return value