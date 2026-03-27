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
from itertools import product


from include.cpg_rbf.cpg_so2 import CPG_SO2
from include.cpg_rbf.rbf import RBF
from include.cpg_rbf.cpg_so2 import CPG_LOCO
from include.cpg_rbf.dual_learner import DIL
from include.lowpass_filter import LPF

LEG_SIDE    = ['R', 'L']
LEG_INDEX   = ["F", "B"]
JOINT_NAMES = [0, 1, 2, 3]

# FOOT_NAMES  = ['R0', 'R1', 'R2', 'L0', 'L1', 'L2']

class DivingBeetleNode(Node):
    def __init__(self):
        super().__init__('diving_beetle_node')
        
        self.rbf = RBF()
        self.cpg = CPG_SO2()

        # It would be centerize the kernel first
        cpg_one_cycle = self.cpg.generate_cpg_one_cycle(0.05)       # lowest phi value
        out0_cpg_one_cycle = cpg_one_cycle['out0_cpg_one_cycle'][:] # Use full cycle of CPG
        out1_cpg_one_cycle = cpg_one_cycle['out1_cpg_one_cycle'][:] # Use full cycle of CPG
        cpg_cycle_length = len(out0_cpg_one_cycle)
        self.rbf.construct_kernels_with_cpg_one_cycle(out0_cpg_one_cycle, out1_cpg_one_cycle, cpg_cycle_length) # construct kernels with cpg one cycle

        # swimming pattern weights
        # self.imitated_weights = np.load('imitated_diving_beetle_swim_forward_weights.npz')
        # self.imitated_weights = np.load('imitated_diving_beetle_swim_backward_weights.npz')
        # self.imitated_weights = np.load('imitated_diving_beetle_swim_turning_weights.npz')
        self.imitated_weights = np.load('imitated_diving_beetle_swim_crossing_weights.npz')



        # ROS2 publishers and subscribers
        self.pub_jcmd                   = self.create_publisher(Float32MultiArray, '/diving_beetle/joint_angle_commands', 1)
        self.pub_expected_force         = self.create_publisher(Float32MultiArray, '/diving_beetle/expected_foot_force', 1)
        self.pub_foot_force             = self.create_publisher(Float32MultiArray, '/diving_beetle/resultant_foot_force_feedback', 1)
        self.pub_foot_force_error       = self.create_publisher(Float32MultiArray, '/diving_beetle/foot_force_error', 1)
        # self.pub_phi_cmd            = self.create_publisher(Float32MultiArray, '/diving_beetle/phi_cmd', 1)
        # self.pub_phi_walk           = self.create_publisher(Float32MultiArray, '/diving_beetle/phi_walk', 1)
        
        self.sub_js                     = self.create_subscription(Joy, '/joy', self.Joy_set, 1)        
        self.sub_terminated_cmd         = self.create_subscription(Bool, '/diving_beetle/terminated_cmd', self.TerminatedCmdCallback, 1)
        self.sub_started_cmd            = self.create_subscription(Bool, '/diving_beetle/started_cmd', self.StartedCmdCallback, 1)
        self.sub_execute_control_cmd    = self.create_subscription(Bool, '/diving_beetle/execute_control_cmd', self.ExecuteControlCmdCallback, 1)
        
        self.sub_foot_force         = self.create_subscription(Float32MultiArray, '/diving_beetle/foot_force_feedback', self.FootForceFeedback, 1)

        # Command variables
        self.initial_joint_angles = {'FR': [ 0,  0, 0, 0],
                                     'BR': [ 0,  0, 0, 0],
                                     'FL': [ 0,  0, 0, 0],
                                     'BL': [ 0,  0, 0, 0]}
        self.initial_joint_angles = np.deg2rad(self.conv2float_arr(self.initial_joint_angles))
        self.joint_angles_cmd = {'FR': [0.0, 0.0, 0.0 ,0.0],
                                 'BR': [0.0, 0.0, 0.0 ,0.0],
                                 'FL': [0.0, 0.0, 0.0 ,0.0],
                                 'BL': [0.0, 0.0, 0.0 ,0.0]}
        
        # Control variables
        self.cpg_modulated = {}
        self.cpg_output = {}
        self.cpg_mod_cmd = {}
        for side in LEG_SIDE:
            for index in LEG_INDEX:
                self.cpg_modulated[f'{index}{side}']     = CPG_LOCO()
                self.cpg_output[f'{index}{side}']        = self.cpg_modulated[f'{index}{side}'].modulate_cpg(0.05, 0.0, 1.0)  # initial run to set up internal states
                self.cpg_mod_cmd[f'{index}{side}']       = {'phi':0.1, 'pause_input': 0.0, 'rewind_input': 1.0}   # initial run to set up internal states
        


        
        # Feedback variables
        self.expected_foot_forces = {}
        self.foot_force_fb = {}
        self.foot_force_lpf = {}
        for side in LEG_SIDE:
            for index in LEG_INDEX:
                self.expected_foot_forces[f'{index}{side}']     = 0.0
                self.foot_force_fb[f'{index}{side}']            = 0.0
                self.foot_force_lpf[f'{index}{side}']           = LPF(0.9)
        
        # ------------------------------ Node Setup ------------------------------
        self.axes_js = np.zeros(8)
        self.btn_js = np.zeros(8)
        
        self.start_flag = False
        self.end_flag = False
        self.execute_control_flag = False
        
        self.start_time = time.time()  # Record the start time
        ros_node_freq = 50 #Hz
        self.timer = self.create_timer((1/ros_node_freq), self.timer_callback)
        
    # =============== Joy Stick ====================
    def Joy_set(self, msg):
        self.axes_js = msg.axes
        self.btn_js = msg.buttons
        # print(self.axes_js)
        # print(self.btn_js)
    # =============== Feedback ====================
    def FootForceFeedback(self, msg):
        ...
        # scaling_factor = 10
        # offset_factor = 0.15
        # self.foot_force_fb['F_R0'] = np.clip(self.lpf_R0.filter(scaling_factor * msg.data[0]) - offset_factor, 0.0, 1.0)  # x, y, z
        # self.foot_force_fb['F_R1'] = np.clip(self.lpf_R1.filter(scaling_factor * msg.data[1]) - offset_factor, 0.0, 1.0)
        # self.foot_force_fb['F_R2'] = np.clip(self.lpf_R2.filter(scaling_factor * msg.data[2]) - offset_factor, 0.0, 1.0)
        # self.foot_force_fb['F_L0'] = np.clip(self.lpf_L0.filter(scaling_factor * msg.data[3]) - offset_factor, 0.0, 1.0)
        # self.foot_force_fb['F_L1'] = np.clip(self.lpf_L1.filter(scaling_factor * msg.data[4]) - offset_factor, 0.0, 1.0)
        # self.foot_force_fb['F_L2'] = np.clip(self.lpf_L2.filter(scaling_factor * msg.data[5]) - offset_factor, 0.0, 1.0)
        # print(self.foot_force_fb)
    # =============== Setup Command ====================
    def StartedCmdCallback(self, msg):
        self.start_flag = msg.data
        # if self.start_flag:
        #     print("Received start command from simulation.")
    def ExecuteControlCmdCallback(self, msg):
        self.execute_control_flag = msg.data
        # if self.execute_control_flag:
        #     print("Received execute control command from simulation.")
    def TerminatedCmdCallback(self, msg):
        self.end_flag = msg.data
        if self.end_flag:
            print("Received termination command from simulation.")
            self.destroy_node()
            rclpy.shutdown()

    # ==============================================
    def conv2float_arr(self,  input_dict):
        # return [float(value) for value in input_dict.values()]
        return [float(angle) for angles in input_dict.values() for angle in angles]
            
    # ==============================================
    def timer_callback(self):
        if self.start_flag:
            self.control_loop()
        
    def control_loop(self):

            
        # ==================== CPF modulation layer ====================  
        for side in LEG_SIDE:
            for index in LEG_INDEX:
                self.cpg_output[f'{index}{side}'] = self.cpg_modulated[f'{index}{side}'].modulate_cpg(self.cpg_mod_cmd[f'{index}{side}']['phi'], 
                                                                                                    self.cpg_mod_cmd[f'{index}{side}']['pause_input'], 
                                                                                                    self.cpg_mod_cmd[f'{index}{side}']['rewind_input'])
        
        # # ==================== RBF layer ====================
        for side in LEG_SIDE:
            for joint in JOINT_NAMES:
                for index in LEG_INDEX:
                    self.joint_angles_cmd[f'{index}{side}'][joint] = self.rbf.regenerate_target_traj(self.cpg_output[f'{index}{side}']['cpg_output_0'], 
                                                                                                    self.cpg_output[f'{index}{side}']['cpg_output_1'],
                                                                                                    self.imitated_weights[f'{index}{side}{joint}'])
        
        if not self.execute_control_flag:
            # set initial posture
            msg_float32 = Float32MultiArray()
            msg_float32.data = [float(x) for x in self.initial_joint_angles]
            self.pub_jcmd.publish(msg_float32)
            
        if self.execute_control_flag:
            msg_float32 = Float32MultiArray()
            
            # joint angle command publish
            msg_float32.data = self.conv2float_arr(self.joint_angles_cmd)
            self.pub_jcmd.publish(msg_float32)



    

def main(args=None):

    rclpy.init(args=args)
    diving_beetle_node = DivingBeetleNode()
    
    try:
        rclpy.spin(diving_beetle_node)
    except (KeyboardInterrupt, ExternalShutdownException):
        print("=============ROS Node [Terminated]=============")

    finally:
        diving_beetle_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()


