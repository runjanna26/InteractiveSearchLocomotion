

robot_pwd = '11223344'

import can
import os
import time
import csv
import traceback
from collections import namedtuple
from enum import Enum
from math import isfinite
import numpy as np
import warnings
from .rmd_motor_can import *


# A class to manage the low level CAN communication protocols
class CAN_Manager(object):
    """A class to manage the low level CAN communication protocols"""
    """
    Set to true to display every message sent and recieved for debugging.
    """
    # Note, defining singletons in this way means that you cannot inherit
    # from this class, as apparently __init__ for the subclass will be called twice
    _instance = None
    """
    Used to keep track of one instantation of the class to make a singleton object
    """
    
    def __new__(cls):
        """
        Makes a singleton object to manage a socketcan_native CAN bus.
        """
        if not cls._instance:
            cls._instance = super(CAN_Manager, cls).__new__(cls)
            print("Initializing CAN_Manager")
            # verify the CAN bus is currently down
            os.system( f'echo {robot_pwd} | sudo -S /sbin/ip link set can0 down' )
            # start the CAN bus back up
            os.system( 'sudo /sbin/ip link set can0 up type can bitrate 1000000' )
            
            # create a python-can bus object
            cls._instance.bus = can.interface.Bus(channel='can0', bustype='socketcan') # socketcan_native / socketcan
            # create a python-can notifier object, which motors can later subscribe to
            cls._instance.notifier = can.Notifier(bus=cls._instance.bus, listeners=[])
            print("Connected on: " + str(cls._instance.bus))

        return cls._instance

    def __init__(self):
        """
        ALl initialization happens in __new__
        """
        pass
        self.debug = False

        
    def __del__(self):
        """
        # shut down the CAN bus when the object is deleted
        # This may not ever get called, so keep a reference and explicitly delete if this is important.
        """
        os.system( 'sudo /sbin/ip link set can0 down' ) 

    # subscribe a motor object to the CAN bus to be updated upon message reception
    def add_motor(self, exo):
        """
        Subscribe a exo object to the CAN bus to be updated upon message reception

        Args:
            self : can manager
            exo: The exo object to be subscribed to the notifier
        """
        self.notifier.add_listener(motorListener(self, exo))

    def send_can_msg(self, message_id, data):
        """
        Sends a message to the exo, with a header of message_id and data array of data

        Args:
            message_id: The CAN ID of the exo to send to.
            data: An array of integers or bytes of data to send.
        """
        DLC = len(data)
        assert (DLC <= 8), ('Data too long in message for motor ' + str(message_id))
        
        if self.debug:
            print('ID: ' + str(hex(message_id)) + '   Data: ' + '[{}]'.format(', '.join(hex(d) for d in data)))
        
        message = can.Message(arbitration_id=message_id, data=data, is_extended_id=False) # Send the message
        time.sleep(0.0001)
        try:
            self.bus.send(message)
            if self.debug:
                print("    Message sent on " + str(self.bus.channel_info) )
        except can.CanError:
            if self.debug:
                print("    Message NOT sent")

    def check_all_motors(self) -> None:
        self.send_can_msg(0x280, [READ_MOTOR_MODEL_ID, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00])            

    def stop_all_motors(self) -> None:
        self.send_can_msg(0x280, [STOP_MOTOR_ID, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00])  

    def shutdown_all_motors(self) -> None:
        self.send_can_msg(0x280, [SHUTDOWN_MOTOR_ID, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00])  

    def reset_all_motors(self) -> None:
        self.send_can_msg(0x280, [REBOOT_MOTOR_ID, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00])

# python-can listener object, with handler to be called upon reception of a message on the CAN bus
class motorListener(can.Listener):
    """Python-can listener object, with handler to be called upon reception of a message on the CAN bus"""
    def __init__(self, canman, dev):
        """
        Sets stores can manager and motor object references
        
        Args:
            canman: The CanManager object to get messages from
            motor: The TMotorCANManager object to update
        """
        self.canman = canman
        self.bus = canman.bus
        self.dev = dev
        print('CAN_Listener initialized ...' )

    def on_message_received(self, msg):
        """
        Updates this listener's motor with the info contained in msg, if that message was for this motor.

        args:
            msg: A python-can CAN message
        """
        data = bytes(msg.data)
        id   = int(msg.arbitration_id)

        if self.canman.debug:
            data_hex = " ".join(f"{b:02X}" for b in data)
            print(f"id: {id:#X}, data: {data_hex}")
        
        if data[0] == READ_MOTOR_MODEL_ID:
            # print("Checking all motors...")
            print(f"All motor in bus [ID: {(id-0x240):#X}]")

        self.dev.parse_feedback_message(id, data)

