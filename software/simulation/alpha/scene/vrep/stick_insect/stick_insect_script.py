
import rclpy
from rclpy.qos import ReliabilityPolicy, QoSProfile
from rclpy.node import Node
from rclpy.executors import ExternalShutdownException
from std_msgs.msg import Float64MultiArray, Float32MultiArray, UInt16MultiArray, Int16MultiArray, Int8, String, UInt16, Int8MultiArray, Bool
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Joy,Imu, Image

import numpy as np
import pandas as pd

class ROS2Interface(Node):
    def __init__(self):
        super().__init__('stick_insect')
        self.sub_q       = self.create_subscription(Float32MultiArray, '/stick_insect/joint_angle_commands', self.JointAnglesCmd, 1)

        self.joint_cmd = {'TR': [],'CR': [],'FR': [],'TL': [],'CL': [],'FL': []}

    def JointAnglesCmd(self, msg):
        self.joint_cmd['TR'] = msg.data[0:3]
        self.joint_cmd['CR'] = msg.data[3:6]
        self.joint_cmd['FR'] = msg.data[6:9]
        self.joint_cmd['TL'] = msg.data[9:12]
        self.joint_cmd['CL'] = msg.data[12:15]
        self.joint_cmd['FL'] = msg.data[15:18]
        # print(self.joint_cmd['FL'])

def sysCall_init():
    rclpy.init()
    sim = require('sim')

    global stick_insect_joint, stick_insect_foot_force


    stick_insect_joint      = {'TR': [],'CR': [],'FR': [],'TL': [],'CL': [],'FL': []}
    stick_insect_foot_force = {'R': [], 'L': []}
    stick_insect_body       = sim.getObject('..')
    for i in range(3):
        stick_insect_joint['TR'].append(sim.getObject('../TR'+str(i)))
        stick_insect_joint['CR'].append(sim.getObject('../CR'+str(i)))
        stick_insect_joint['FR'].append(sim.getObject('../FR'+str(i)))
        stick_insect_joint['TL'].append(sim.getObject('../TL'+str(i)))
        stick_insect_joint['CL'].append(sim.getObject('../CL'+str(i)))
        stick_insect_joint['FL'].append(sim.getObject('../FL'+str(i)))
        stick_insect_foot_force['R'].append(sim.getObject('../R'+str(i)+'_fs'))
        stick_insect_foot_force['L'].append(sim.getObject('../L'+str(i)+'_fs'))
    # print(stick_insect_joint)
    # print(stick_insect_foot_force)
    

    # sim.setObjectInt32Parameter(stick_insect_body, sim.shapeintparam_static, 0) 

    self.ros2_node = ROS2Interface()

     

def sysCall_actuation():
    pass
    # print(self.ros2_node.joint_cmd)
    
    for group, angles in self.ros2_node.joint_cmd.items():   # Set joint target positions using a loop
        for i, angle in enumerate(angles):
            sim.setJointTargetPosition(stick_insect_joint[group][i], angle) # rad


def sysCall_sensing():
    pass
    rclpy.spin_once(self.ros2_node, timeout_sec=0)

def sysCall_cleanup():
    pass

#======================================================================================
def rotx(a):
    a = np.radians(a)
    return np.matrix([[1,   0,     0,  0],
                      [0, c(a), -s(a),  0],
                      [0, s(a),  c(a),  0],
                      [0,    0,     0,  1]])
def c(x):
    return np.cos(x)
def s(x):
    return np.sin(x)