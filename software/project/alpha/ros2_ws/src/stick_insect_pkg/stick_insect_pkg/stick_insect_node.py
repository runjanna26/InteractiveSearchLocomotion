import rclpy
from rclpy.qos import ReliabilityPolicy, QoSProfile
from rclpy.node import Node
from rclpy.executors import ExternalShutdownException
from std_msgs.msg import Float64MultiArray, Float32MultiArray, UInt16MultiArray, Int16MultiArray, Int8, String, UInt16, Int8MultiArray, Bool
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Joy,Imu, Image
from std_srvs.srv import SetBool, Trigger

import numpy as np
import time

class StickInsectNode(Node):
    def __init__(self):
        super().__init__('stick_insect_node')

        self.pub_jcmd     = self.create_publisher(Float32MultiArray, '/stick_insect/joint_angle_commands', 1)


        self.initial_joint_angles = {'TR': [30,    0, -40],
                                     'CR': [10,    0,  10],
                                     'FR': [-60, -60, -60],
                                     'TL': [ 30,   0, -40],
                                     'CL': [ 10,   0,  10],
                                     'FL': [-60, -60, -60]}
        self.initial_joint_angles = np.deg2rad(self.conv2float_arr(self.initial_joint_angles))
        

        self.start_time = time.time()  # Record the start time
        ros_node_freq = 100 #Hz
        self.timer = self.create_timer((1/ros_node_freq), self.timer_callback)

    def timer_callback(self):
        ...
        if time.time() - self.start_time < 2:
            q_msg = Float32MultiArray()
            q_msg.data = [float(x) for x in self.initial_joint_angles]
            self.pub_jcmd.publish(q_msg)
        # else:
            # self.q_cmd = [1.0, 2.0, 3.0,
            #             1.0, 2.0, 3.0,
            #             1.0, 2.0, 3.0,
            #             1.0, 2.0, 3.0,
            #             1.0, 2.0, 3.0,
            #             1.0, 2.0, 3.0]
            # q_msg = Float32MultiArray()
            # q_msg.data = [float(x) for x in self.q_cmd]
            # self.pub_jcmd.publish(q_msg)

    def conv2float_arr(self,  input_dict):
        # return [float(value) for value in input_dict.values()]
        return [float(angle) for angles in input_dict.values() for angle in angles]
    

def main(args=None):

    rclpy.init(args=args)
    stick_insect_node = StickInsectNode()
    
    try:
        rclpy.spin(stick_insect_node)
    except (KeyboardInterrupt, ExternalShutdownException):
        print("=============ROS Node [Terminated]=============")

    finally:
        stick_insect_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()


