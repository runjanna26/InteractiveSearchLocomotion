'''
Credit: https://github.com/neurobionics/TMotorCANControl
[0.1] Install can-utils on linux
$ sudo apt-get intsall can-utils
[0.2] Install python-can
$ sudo apt update
$ pip3 install python-can

[1] Upload firmware with ST-Link v.2 by STM32Programmer:
candleLight firmware (canable2_fw-ba6b1dd.bin)

[2] check canable board was connected with Canable v2.0 firmware
$ lsusb

[3] Set bitrate 1Mbps
$ sudo ip link set can0 type can bitrate 1000000
[4] Set can0 up
$ sudo ip link set can0 up
[5] Check status can0
$ ip a



# [] Install SocketCAN driver
# $ make clean
# $ make NET=NETDEV_SUPPORT
# $ sudo make install
# $ sudo modprobe pcan
'''

robot_pwd = 'exvis123'

import can
import time
import csv
import traceback
from collections import namedtuple
from enum import Enum
from math import isfinite
import numpy as np
import warnings

import os
from collections import namedtuple
from math import isfinite

from typing import List

MIT         = 0x400         # MIT control command
SINGLE      = 0x140      # single motor control command
BOARDCAST   = 0x280      # multi motor control command

READ_MULTI_TURN_OUTPUT_SHAFT_ANGLE_ID   = 0x60   
READ_SINGLE_TURN_OUTPUT_SHAFT_ANGLE_ID  = 0x94
READ_MOTOR_STATUS_1_ID                  = 0x9A
READ_MOTOR_STATUS_2_ID                  = 0x9C
READ_MOTOR_STATUS_3_ID                  = 0x9D
READ_MOTOR_MODEL_ID                     = 0xB5
READ_ACCELATION_ID                      = 0x42

STOP_MOTOR_ID                   = 0x81
SHUTDOWN_MOTOR_ID               = 0x80
REBOOT_MOTOR_ID                 = 0x76

CHANGE_ID_MOTOR_ID              = 0x79
CAN_ID_READ_MOTOR_ID            = 0x01
CAN_ID_WRITE_MOTOR_ID           = 0x00


motor_param = {
                'X4-10': {
                    'max_torque':   10.0, 
                    'max_speed':    25.0, 
                    'max_current':  7.8,  
                    'Kt': 0.85              # Torque constant, Nm/A
                    } 
              }

ERROR = {'MotorStall':              0x0002,
         'LowVoltage':              0x0004,
         'OverVoltage':             0x0008,
         'OverCurrent':             0x0010,
         'PowerOverrun':            0x0040,
         'CalibrationError':        0x0080,
         'OverSpeed':               0x0100,
         'OverTemperature':         0x0800,
         'OverTe,mperature_Motor':  0x1000,
         'EncoderError':            0x2000,
         'EncoderDataError':        0x4000}

class rmd_motor_can():
    def __init__(self, motor_id, can_manager) -> None:
        self.debug = False
        self.print_feedback = False

        self.motor_id = motor_id 
        if self.debug:
            ...
        else:
            self.can_manager = can_manager
            self.type = 'X4-10'
            self.can_manager.add_motor(self)

        # Control variables
        self.desired_position               = 0.0   # rad
        self.desired_velocity               = 0.0   # rad/s
        self.desired_torque                 = 0.0   # Nm
        self.Kp                             = 0.0
        self.Kd                             = 0.0

        # Feedback variables
        self.feedback_position              = 0.0
        self.feedback_multi_turn_position   = 0.0
        self.feedback_velocity              = 0.0
        self.feedback_motor_torque          = 0.0

        self.feedback_temperature           = 0.0
        self.feedback_mosfet_temperature    = 0.0

        self.feedback_voltage               = 0.0
        self.feedback_current               = 0.0
        
        self.error_state                    = False
        self.errors                         = []






    def change_id(self, new_id) -> None:
        new_id = int(new_id)

        if new_id < 0 or new_id > 32:
            raise ValueError("ID must be between 0 and 32")
        
        self.can_manager.send_can_msg(SINGLE + self.motor_id, [CHANGE_ID_MOTOR_ID, 0x00, CAN_ID_WRITE_MOTOR_ID, 0x00, 0x00, 0x00, 0x00, new_id])
        time.sleep(0.1)
        self.motor_id = new_id
        
        print(f"Motor ID changed to {self.motor_id}")
        self.reboot()

        self.can_manager.send_can_msg(SINGLE + self.motor_id, [CHANGE_ID_MOTOR_ID, 0x00, CAN_ID_READ_MOTOR_ID, 0x00, 0x00, 0x00, 0x00, 0x00])


    def shutdown(self) -> None:
        '''Diasble motor by free motor'''
        self.can_manager.send_can_msg(SINGLE + self.motor_id, [SHUTDOWN_MOTOR_ID, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00])
    def stop(self) -> None:
        '''Stop motor by holding latest position'''
        self.can_manager.send_can_msg(SINGLE + self.motor_id, [STOP_MOTOR_ID, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00])
    def reboot(self) -> None:
        '''Reset motor'''
        self.can_manager.send_can_msg(SINGLE + self.motor_id, [REBOOT_MOTOR_ID, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00])


    # send an MIT control signal, consisting of desired position, velocity, and current, and gains for position and velocity control
    # basically an impedance controller
    def MIT_controller(self, position, velocity, Kp, Kd, Tff):
        """
        Sends an MIT style control signal to the motor. This signal will be used to generate a 
        current for the field-oriented controller on the motor control chip, given by this expression:

            q_control = Kp*(position - current_position) + Kd*(velocity - current_velocity) + Tff

        Args:
            motor_id: The CAN ID of the motor to send the message to
            motor_type: A string noting the type of motor, ie 'AK80-9'
            position: The desired position in rad
            velocity: The desired velocity in rad/s
            Kp: The position gain
            Kd: The velocity gain
            Tff: The additional current
        """
        position_uint16     = self.float_to_uint(position, -12.5, 12.5, 16)
        velocity_uint12     = self.float_to_uint(velocity, -45, 45, 12)
        Kp_uint12           = self.float_to_uint(Kp, 0, 500, 12)
        Kd_uint12           = self.float_to_uint(Kd, 0, 5, 12)
        Tff_uint12          = self.float_to_uint(Tff, -24, 24, 12)

        data = [
            position_uint16 >> 8,
            position_uint16 & 0x00FF,
            (velocity_uint12) >> 4,
            ((velocity_uint12&0x00F)<<4) | (Kp_uint12) >> 8,
            (Kp_uint12&0x0FF),
            (Kd_uint12) >> 4,
            ((Kd_uint12&0x00F)<<4) | (Tff_uint12) >> 8,
            (Tff_uint12&0x0FF)
        ]
        self.can_manager.send_can_msg(0x400 + self.motor_id, data)

    def update(self):


        if np.abs(self.desired_velocity) > motor_param[self.type]['max_speed']:
            print('Velocity command exceeds maximum speed limit')

        # if np.abs(self.desired_torque) > motor_param[self.type]['max_torque']:
        #     print('Torque command exceeds maximum torque limit')

        self.desired_velocity   = self.limit_value(self.desired_velocity, -motor_param[self.type]['max_speed'], motor_param[self.type]['max_speed'])
        # self.desired_torque     = self.limit_value(self.desired_torque, -motor_param[self.type]['max_torque'], motor_param[self.type]['max_torque'])


        # Send control command
        self.MIT_controller(self.desired_position, 
                            self.desired_velocity, 
                            self.Kp, 
                            self.Kd, 
                            self.desired_torque)
        # Request feedback
        self.can_manager.send_can_msg(SINGLE + self.motor_id, [READ_SINGLE_TURN_OUTPUT_SHAFT_ANGLE_ID, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00])
        self.can_manager.send_can_msg(SINGLE + self.motor_id, [READ_MULTI_TURN_OUTPUT_SHAFT_ANGLE_ID, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00])
        self.can_manager.send_can_msg(SINGLE + self.motor_id, [READ_MOTOR_STATUS_1_ID, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00])
        self.can_manager.send_can_msg(SINGLE + self.motor_id, [READ_MOTOR_STATUS_2_ID, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]) 

    # ================== Listen message ================= #
    def parse_feedback_message(self, id, data):
        if len(data) != 8:
            if self.debug:
                print("Invalid data length")
            return

        if id == 0x300:
            if data[0] == CHANGE_ID_MOTOR_ID:
                new_id = data[7] << 8 | data[6]
                print(f"Now Motor ID have already changed to {new_id} (self.motor_id: {self.motor_id})")


        if id == (0x240 + self.motor_id):  
            if data[0] == READ_SINGLE_TURN_OUTPUT_SHAFT_ANGLE_ID:
                raw = (data[6]) | (data[7]<<8)
                # if raw & 0x8000:  # signed 16-bit
                #     raw -= 0x10000
                self.feedback_position = raw * 0.01 * np.pi / 180.0
                if self.print_feedback:
                    print(f"Motor ID: {self.motor_id} | Position: {self.feedback_position :.2f} rad")

            if data[0] == READ_MULTI_TURN_OUTPUT_SHAFT_ANGLE_ID: 
                raw = (data[7] << 24 | data[6] << 16 | data[5] << 8 | data[4])
                if raw & 0x80000000:
                    raw -= 0x100000000
                self.feedback_multi_turn_position = raw * (2 * np.pi) / (2**18)  # Convert to radians
                if self.print_feedback:
                    print(f"Motor ID: {self.motor_id} | Multi-Turn Position: {self.feedback_multi_turn_position :.2f} rad")

            if data[0] == READ_MOTOR_STATUS_2_ID:
                temperature             = data[1]
                torque_current_motor    = (data[2] | (data[3] << 8))
                velocity                = (data[4] | (data[5] << 8))
                position                = (data[6] | (data[7] << 8))
                if torque_current_motor & 0x8000:
                    torque_current_motor -= 0x10000
                if velocity & 0x8000:
                    velocity -= 0x10000
                if position & 0x8000:
                    position -= 0x10000
                self.feedback_temperature    = float(temperature)  # Celsius
                self.feedback_current        = torque_current_motor * 0.01  # Amps
                self.feedback_motor_torque   = torque_current_motor * 0.01 * motor_param[self.type]['Kt']  # Nm
                self.feedback_velocity       = velocity * np.pi / 180.0  # rad/s
                if self.print_feedback:
                    print(f"Motor ID: {self.motor_id} | Temp: {temperature} C | Torque: {self.feedback_motor_torque:.3f} Nm | Velocity: {self.feedback_velocity:.2f} rad/s")
            
            if data[0] == READ_MOTOR_STATUS_1_ID:
                error_list = []
                MOSFET_temperature  = data[2]
                voltage             = (data[4] | (data[5] << 8))
                error               = data[6] | (data[7] << 8) 
                
                if error != 0:
                    self.error_state = True
                    error_list = [name for name, bit in ERROR.items() if error & bit]
                    if self.print_feedback:
                        print(f"Motor ID: {self.motor_id} | Errors: {', '.join(error_list)}")
                else:
                    self.error_state = False
                self.errors             = error_list
                self.feedback_voltage           = voltage * 0.1  # Volts
                self.feedback_mosfet_temperature = float(MOSFET_temperature)  # Celsius
                if self.print_feedback:
                    print(f"Motor ID: {self.motor_id} | Voltage: {self.voltage:.1f} V | MOSFET Temp: {self.feedback_mosfet_temperature} C")
                
    
    def set_desired_position_radian(self, position):
        self.desired_position = float(position)
    def set_desired_position_degree(self, position):
        self.desired_position = float(position * np.pi / 180.0)  

    def set_desired_velocity_radian_per_second(self, velocity):
        self.desired_velocity = float(velocity)
    def set_desired_velocity_degree_per_second(self, velocity):
        self.desired_velocity = float(velocity * np.pi / 180.0)

    def set_desired_torque(self, torque):
        self.desired_torque = torque
    
    def set_desired_stiffness(self, Kp):
        self.Kp = Kp
    def set_desired_damping(self, Kd):
        self.Kd = Kd



    # ================= Utility functions ================ #
    def float_to_uint(self,x,x_min,x_max,num_bits):
        """
        Interpolates a floating point number to an unsigned integer of num_bits length.
        A number of x_max will be the largest integer of num_bits, and x_min would be 0.

        args:
            x: The floating point number to convert
            x_min: The minimum value for the floating point number
            x_max: The maximum value for the floating point number
            num_bits: The number of bits for the unsigned integer
        """
        span = x_max-x_min
        bitratio = float((1<<num_bits)/span)
        x = self.limit_value(x,x_min,x_max-(2/bitratio))
        return self.limit_value(int((x- x_min)*( bitratio )),0,int((x_max-x_min)*bitratio) )

    def uint_to_float(self, x,x_min,x_max,num_bits):
        """
        Interpolates an unsigned integer of num_bits length to a floating point number between x_min and x_max.

        args:
            x: The floating point number to convert
            x_min: The minimum value for the floating point number
            x_max: The maximum value for the floating point number
            num_bits: The number of bits for the unsigned integer
        """
        span = x_max-x_min
        return float(x*span/((1<<num_bits)-1) + x_min)
    
    # Locks value between min and max
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