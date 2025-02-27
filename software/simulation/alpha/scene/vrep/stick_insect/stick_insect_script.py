
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
        self.sub_q_cmd      = self.create_subscription(Float32MultiArray, '/stick_insect/joint_angle_commands', self.JointAnglesCmd, 1)
        self.pub_q_fb       = self.create_publisher(Float32MultiArray, '/stick_insect/joint_angle_feedback', 1)
        self.pub_foot_force = self.create_publisher(Float32MultiArray, '/stick_insect/foot_force_feedback', 1)
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
    global sim
    sim = require('sim')

    global stick_insect_joint_obj, stick_insect_foot_force_obj, stick_insect_body_obj

    # Objects
    stick_insect_joint_obj      = {'TR': [],'CR': [],'FR': [],'TL': [],'CL': [],'FL': []}
    stick_insect_foot_force_obj = {'R': [], 'L': []}
    stick_insect_body_obj       = sim.getObject('..')
    for i in range(3):
        stick_insect_joint_obj['TR'].append(sim.getObject('../TR'+str(i)))
        stick_insect_joint_obj['CR'].append(sim.getObject('../CR'+str(i)))
        stick_insect_joint_obj['FR'].append(sim.getObject('../FR'+str(i)))
        stick_insect_joint_obj['TL'].append(sim.getObject('../TL'+str(i)))
        stick_insect_joint_obj['CL'].append(sim.getObject('../CL'+str(i)))
        stick_insect_joint_obj['FL'].append(sim.getObject('../FL'+str(i)))
        stick_insect_foot_force_obj['R'].append(sim.getObject('../R'+str(i)+'_fs'))
        stick_insect_foot_force_obj['L'].append(sim.getObject('../L'+str(i)+'_fs'))
    # print(stick_insect_joint_obj)
    # print(stick_insect_foot_force_obj)
    
    global stick_insect_joint_fb
    stick_insect_joint_fb      = {'TR': [0.0 ,0.0 ,0.0],
                                  'CR': [0.0 ,0.0 ,0.0],
                                  'FR': [0.0 ,0.0 ,0.0],
                                  'TL': [0.0 ,0.0 ,0.0],
                                  'CL': [0.0 ,0.0 ,0.0],
                                  'FL': [0.0 ,0.0 ,0.0]}
    global stick_insect_foot_force_fb
    stick_insect_foot_force_fb = {'R': [0.0, 0.0, 0.0], 
                                  'L': [0.0, 0.0, 0.0]}

    # Setup simulation
    # sim.setObjectInt32Parameter(stick_insect_body_obj, sim.shapeintparam_static, 0) 

    self.ros2_node = ROS2Interface()

     

def sysCall_actuation():
    pass
    # print(self.ros2_node.joint_cmd)
    
    for group, angles in self.ros2_node.joint_cmd.items():   # Set joint target positions using a loop
        for i, angle in enumerate(angles):
            sim.setJointTargetPosition(stick_insect_joint_obj[group][i], angle) # rad
    
        

def sysCall_sensing():
    pass
    
    # ================ publish joint feedback ================
    for group in stick_insect_joint_fb:
        stick_insect_joint_fb[group][0] = sim.getJointPosition(stick_insect_joint_obj[group][0]) # rad
        stick_insect_joint_fb[group][1] = sim.getJointPosition(stick_insect_joint_obj[group][1]) # rad    
        stick_insect_joint_fb[group][2] = sim.getJointPosition(stick_insect_joint_obj[group][2]) # rad
    # print(stick_insect_joint_fb)
    msg_float32 = Float32MultiArray()
    msg_float32.data = conv2float_arr(stick_insect_joint_fb)
    self.ros2_node.pub_q_fb.publish(msg_float32)
    # ================ publish foot force feedback ================
    for group in stick_insect_foot_force_fb:
        _, stick_insect_foot_force_fb[group], _ = sim.readForceSensor(stick_insect_foot_force_obj[group][0])
        _, stick_insect_foot_force_fb[group], _ = sim.readForceSensor(stick_insect_foot_force_obj[group][1])
        _, stick_insect_foot_force_fb[group], _ = sim.readForceSensor(stick_insect_foot_force_obj[group][2])
    # print(stick_insect_foot_force_fb)
    msg_float32 = Float32MultiArray()
    msg_float32.data = conv2float_arr(stick_insect_foot_force_fb)
    self.ros2_node.pub_foot_force.publish(msg_float32)    
        
    # ================ spin ros node ================
    rclpy.spin_once(self.ros2_node, timeout_sec=0)

def sysCall_cleanup():
    pass

#======================================================================================
def conv2float_arr(input_dict):
    # return [float(value) for value in input_dict.values()]
    return [float(angle) for angles in input_dict.values() for angle in angles]
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