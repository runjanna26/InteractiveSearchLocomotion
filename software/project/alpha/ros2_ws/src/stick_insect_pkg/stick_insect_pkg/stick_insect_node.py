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


from include.cpg_rbf.cpg_so2 import CPG_SO2
from include.cpg_rbf.rbf import RBF
from include.cpg_rbf.cpg_so2 import CPG_LOCO



class StickInsectNode(Node):
    def __init__(self):
        super().__init__('stick_insect_node')
        
        self.rbf = RBF()
        self.cpg = CPG_SO2()
        self.cpg_loco = CPG_LOCO()

        # It would be centerize the kernel first
        cpg_one_cycle = self.cpg.generate_cpg_one_cycle(0.05)    # lowest phi value
        out0_cpg_one_cycle = cpg_one_cycle['out0_cpg_one_cycle'][:] # Use full cycle of CPG
        out1_cpg_one_cycle = cpg_one_cycle['out1_cpg_one_cycle'][:] # Use full cycle of CPG
        cpg_cycle_length = len(out0_cpg_one_cycle)
        self.rbf.construct_kernels_with_cpg_one_cycle(out0_cpg_one_cycle, out1_cpg_one_cycle, cpg_cycle_length) # construct kernels with cpg one cycle


        self.imitated_weights = np.load('imitated_stick_insect_right_leg_weights.npz')
        

        self.pub_jcmd     = self.create_publisher(Float32MultiArray, '/stick_insect/joint_angle_commands', 1)
        self.sub_js = self.create_subscription(Joy, '/joy', self.Joy_set, 1)


        self.initial_joint_angles = {'TR': [30,    0, -40],
                                     'CR': [10,    0,  10],
                                     'FR': [-60, -60, -60],
                                     'TL': [ 30,   0, -40],
                                     'CL': [ 10,   0,  10],
                                     'FL': [-60, -60, -60]}
        self.initial_joint_angles = np.deg2rad(self.conv2float_arr(self.initial_joint_angles))
        

        self.joint_angles = {'TR': [0.0, 0.0 ,0.0],
                             'CR': [0.0, 0.0 ,0.0],
                             'FR': [0.0, 0.0 ,0.0],
                             'TL': [0.0, 0.0 ,0.0],
                             'CL': [0.0, 0.0 ,0.0],
                             'FL': [0.0, 0.0 ,0.0]}

        self.start_time = time.time()  # Record the start time
        ros_node_freq = 50 #Hz
        self.timer = self.create_timer((1/ros_node_freq), self.timer_callback)
        
        self.cpg_output = self.cpg_loco.modulate_cpg(0.05, 0.0, 1.0)
        self.axes_js = []
        self.btn_js = []

    # =============== Joy Stick ====================
    def Joy_set(self, msg):
        self.axes_js = msg.axes
        self.btn_js = msg.buttons  
        # print(self.axes_js)
        # print(self.btn_js)
        
    def conv2float_arr(self,  input_dict):
        # return [float(value) for value in input_dict.values()]
        return [float(angle) for angles in input_dict.values() for angle in angles]

    def timer_callback(self):
        ...
        
        if self.btn_js[0] == 1: # A button
            self.cpg_output = self.cpg_loco.modulate_cpg(0.05, 0.0, 1.0)
        if self.btn_js[1] == 1: # B button
            self.cpg_output = self.cpg_loco.modulate_cpg(0.03, 0.0, 1.0)    
        if self.btn_js[2] == 1: # X button
            self.cpg_output = self.cpg_loco.modulate_cpg(0.1, 0.0, 1.0)    
            
        
        self.joint_angles['TR'][0] = self.rbf.regenerate_target_traj(self.cpg_output['cpg_output_0'], self.cpg_output['cpg_output_1'], self.imitated_weights['TR0'])         
        self.joint_angles['CR'][0] = self.rbf.regenerate_target_traj(self.cpg_output['cpg_output_0'], self.cpg_output['cpg_output_1'], self.imitated_weights['CR0'])
        self.joint_angles['FR'][0] = self.rbf.regenerate_target_traj(self.cpg_output['cpg_output_0'], self.cpg_output['cpg_output_1'], self.imitated_weights['FR0'])
        
        self.joint_angles['TR'][1] = self.rbf.regenerate_target_traj(self.cpg_output['cpg_output_0'], self.cpg_output['cpg_output_1'], self.imitated_weights['TR1'])         
        self.joint_angles['CR'][1] = self.rbf.regenerate_target_traj(self.cpg_output['cpg_output_0'], self.cpg_output['cpg_output_1'], self.imitated_weights['CR1'])
        self.joint_angles['FR'][1] = self.rbf.regenerate_target_traj(self.cpg_output['cpg_output_0'], self.cpg_output['cpg_output_1'], self.imitated_weights['FR1'])
        
        self.joint_angles['TR'][2] = self.rbf.regenerate_target_traj(self.cpg_output['cpg_output_0'], self.cpg_output['cpg_output_1'], self.imitated_weights['TR2'])         
        self.joint_angles['CR'][2] = self.rbf.regenerate_target_traj(self.cpg_output['cpg_output_0'], self.cpg_output['cpg_output_1'], self.imitated_weights['CR2'])
        self.joint_angles['FR'][2] = self.rbf.regenerate_target_traj(self.cpg_output['cpg_output_0'], self.cpg_output['cpg_output_1'], self.imitated_weights['FR2'])
        
        self.joint_angles['TL'][0] = self.rbf.regenerate_target_traj(self.cpg_output['cpg_output_0'], self.cpg_output['cpg_output_1'], self.imitated_weights['TR0'])         
        self.joint_angles['CL'][0] = self.rbf.regenerate_target_traj(self.cpg_output['cpg_output_0'], self.cpg_output['cpg_output_1'], self.imitated_weights['CR0'])
        self.joint_angles['FL'][0] = self.rbf.regenerate_target_traj(self.cpg_output['cpg_output_0'], self.cpg_output['cpg_output_1'], self.imitated_weights['FR0'])
        
        self.joint_angles['TL'][1] = self.rbf.regenerate_target_traj(self.cpg_output['cpg_output_0'], self.cpg_output['cpg_output_1'], self.imitated_weights['TR1'])         
        self.joint_angles['CL'][1] = self.rbf.regenerate_target_traj(self.cpg_output['cpg_output_0'], self.cpg_output['cpg_output_1'], self.imitated_weights['CR1'])
        self.joint_angles['FL'][1] = self.rbf.regenerate_target_traj(self.cpg_output['cpg_output_0'], self.cpg_output['cpg_output_1'], self.imitated_weights['FR1'])
        
        self.joint_angles['TL'][2] = self.rbf.regenerate_target_traj(self.cpg_output['cpg_output_0'], self.cpg_output['cpg_output_1'], self.imitated_weights['TR2'])         
        self.joint_angles['CL'][2] = self.rbf.regenerate_target_traj(self.cpg_output['cpg_output_0'], self.cpg_output['cpg_output_1'], self.imitated_weights['CR2'])
        self.joint_angles['FL'][2] = self.rbf.regenerate_target_traj(self.cpg_output['cpg_output_0'], self.cpg_output['cpg_output_1'], self.imitated_weights['FR2'])
        
        if time.time() - self.start_time < 2:
            q_msg = Float32MultiArray()
            q_msg.data = [float(x) for x in self.initial_joint_angles]
            self.pub_jcmd.publish(q_msg)
        else:
            self.q_cmd = self.conv2float_arr(self.joint_angles)
            q_msg = Float32MultiArray()
            q_msg.data = [float(x) for x in self.q_cmd]
            self.pub_jcmd.publish(q_msg)


    

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


