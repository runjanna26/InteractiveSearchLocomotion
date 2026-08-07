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

robot_pwd = '11223344'

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
        'P_MIN': -12.5, 'P_MAX': 12.5,
        'V_MIN': -45.0, 'V_MAX': 45.0,
        'T_MIN': -24.0, 'T_MAX': 24.0,
        'Kp_MIN': 0.0,  'Kp_MAX': 500.0,
        'Kd_MIN': 0.0,  'Kd_MAX': 5.0,
        'Kt': 0.85
    },
    'X4-36': {
        'P_MIN': -12.5, 'P_MAX': 12.5,
        'V_MIN': -2.5,  'V_MAX': 2.5,
        'T_MIN': -34.0, 'T_MAX': 34.0,
        'Kp_MIN': 0.0,  'Kp_MAX': 500.0,
        'Kd_MIN': 0.0,  'Kd_MAX': 5.0,
        'Kt': 1.9
    },
    'X4-36_driven': {
        'P_MIN': -12.5, 'P_MAX': 12.5,
        'V_MIN': -45.0, 'V_MAX': 45.0,
        'T_MIN': -34.0, 'T_MAX': 34.0,
        'Kp_MIN': 0.0,  'Kp_MAX': 500.0,
        'Kd_MIN': 0.0,  'Kd_MAX': 5.0,
        'Kt': 1.9
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
    def __init__(self, motor_id, can_manager, motor_type='X4-36') -> None:
        self.debug = False
        self.print_feedback = False

        self.motor_id = motor_id 
        if self.debug:
            ...
        else:
            self.can_manager = can_manager
            self.type = motor_type # FIX: Set type dynamically instead of hardcoding 'X4-10'
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
        # Load the configuration for the current motor type
        cfg = motor_param[self.type]
        # Use the dictionary bounds dynamically
        position_uint16 = self.float_to_uint(position, cfg['P_MIN'],  cfg['P_MAX'],  16)
        velocity_uint12 = self.float_to_uint(velocity, cfg['V_MIN'],  cfg['V_MAX'],  12)
        Kp_uint12       = self.float_to_uint(Kp,       cfg['Kp_MIN'], cfg['Kp_MAX'], 12)
        Kd_uint12       = self.float_to_uint(Kd,       cfg['Kd_MIN'], cfg['Kd_MAX'], 12)
        Tff_uint12      = self.float_to_uint(Tff,      cfg['T_MIN'],  cfg['T_MAX'],  12)

        # Fully explicit masking to prevent bit-bleed
        data = [
            (position_uint16 >> 8) & 0xFF,
            position_uint16 & 0xFF,
            (velocity_uint12 >> 4) & 0xFF,
            ((velocity_uint12 & 0x0F) << 4) | ((Kp_uint12 >> 8) & 0x0F),
            Kp_uint12 & 0xFF,
            (Kd_uint12 >> 4) & 0xFF,
            ((Kd_uint12 & 0x0F) << 4) | ((Tff_uint12 >> 8) & 0x0F),
            Tff_uint12 & 0xFF
        ]
        self.can_manager.send_can_msg(0x400 + self.motor_id, data)

    def send_motor_velocity(self, vel_des):
        """
        Sends a standard RMD velocity command (0xA2).
        Mirrors C++ AKMotor::send_motor_velocity
        """
        # Convert rad/s to degrees/s, multiplied by 100 as per C++
        v_int = int(vel_des * 57.29578 * 100)
        
        # Handle Python negative numbers for bitwise operations (32-bit signed int)
        if v_int < 0:
            v_int = (1 << 32) + v_int

        data = [
            0xA2, 
            0xFF, # Match C++ message.data[1] = 0xFF;
            0x00, 
            0x00,
            v_int & 0xFF,
            (v_int >> 8) & 0xFF,
            (v_int >> 16) & 0xFF,
            (v_int >> 24) & 0xFF
        ]
        
        # Send on standard ID (0x140 + motor_id)
        self.can_manager.send_can_msg(0x140 + self.motor_id, data)

    def send_motor_position(self, pos_des, vel_max=45.0):
        """
        Sends a standard RMD position command (0xA4) with speed limit.
        Mirrors C++ AKMotor::send_motor_position
        """
        # Convert rads to degrees * 100
        p_int = int(pos_des * 57.2957795 * 100)
        
        # Handle Python negative numbers (32-bit signed int)
        if p_int < 0:
            p_int = (1 << 32) + p_int
            
        # Convert max rad/s to degrees/s
        v_max_int = int(vel_max * 57.2957795)
        
        data = [
            0xA4, 
            0x00,
            v_max_int & 0xFF,
            (v_max_int >> 8) & 0xFF,
            p_int & 0xFF,
            (p_int >> 8) & 0xFF,
            (p_int >> 16) & 0xFF,
            (p_int >> 24) & 0xFF
        ]
        
        self.can_manager.send_can_msg(0x140 + self.motor_id, data)
        
    def send_motor_torque(self, torque_des):
        """
        Sends a standard RMD torque/current command (0xA1).
        This replaces the MIT controller for MyActuator motors.
        """
        # 1. Limit the requested torque to the motor's safety bounds
        torque_des = self.limit_value(
            torque_des, 
            motor_param[self.type]['T_MIN'], 
            motor_param[self.type]['T_MAX']
        )
        
        # 2. Convert torque (Nm) to current (Amps) using the Torque Constant (Kt)
        current_des = torque_des / motor_param[self.type]['Kt']
        
        # 3. Convert current to RMD's expected 0.01A integer scale
        iq_int = int(current_des * 100)
        
        # 4. Handle Python negative numbers for 16-bit signed integer packing
        if iq_int < 0:
            iq_int = (1 << 16) + iq_int
            
        data = [
            0xA1, 
            0x00, 
            0x00, 
            0x00, 
            iq_int & 0xFF, 
            (iq_int >> 8) & 0xFF, 
            0x00, 
            0x00
        ]
        
        self.can_manager.send_can_msg(0x140 + self.motor_id, data)
        
    def update(self):
        if np.abs(self.desired_velocity) > motor_param[self.type]['V_MAX']:
            print('Velocity command exceeds maximum speed limit')

        self.desired_velocity = self.limit_value(
            self.desired_velocity, 
            motor_param[self.type]['V_MIN'], 
            motor_param[self.type]['V_MAX']
        )

        # Send control command
        # self.MIT_controller(self.desired_position, 
        #                     self.desired_velocity, 
        #                     self.Kp, 
        #                     self.Kd, 
        #                     self.desired_torque)
        
        # self.send_motor_velocity(self.desired_velocity)
        
        self.send_motor_torque(self.desired_torque)
                            
        # FIX: Remove these! Polling 0x140 while in MIT mode will glitch the motor's state machine.
        self.can_manager.send_can_msg(SINGLE + self.motor_id, [READ_SINGLE_TURN_OUTPUT_SHAFT_ANGLE_ID,  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00])
        self.can_manager.send_can_msg(SINGLE + self.motor_id, [READ_MULTI_TURN_OUTPUT_SHAFT_ANGLE_ID,  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00])
        self.can_manager.send_can_msg(SINGLE + self.motor_id, [READ_MOTOR_STATUS_1_ID,  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00])
        self.can_manager.send_can_msg(SINGLE + self.motor_id, [READ_MOTOR_STATUS_2_ID,  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00])

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
    def float_to_uint(self, x, x_min, x_max, bits):
        """
        Exactly mirrors the C++ AKMotor::float_to_uint function.
        We intentionally use (1 << bits) instead of (1 << bits) - 1, 
        and int() truncation to match the C++ firmware quirk.
        """
        span = float(x_max - x_min)
        
        # Clamp bounds
        if x < x_min:
            x = x_min
        elif x > x_max:
            x = x_max
            
        # Match C++ exact math: (x - x_min) * (float(1 << bits) / span)
        # int() ensures we truncate the decimal just like a C++ (uint16_t) cast
        return int((x - x_min) * (float(1 << bits) / span))

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