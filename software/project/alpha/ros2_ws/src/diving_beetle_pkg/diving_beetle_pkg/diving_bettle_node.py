import queue

import rclpy
from rclpy.qos import ReliabilityPolicy, QoSProfile
from rclpy.node import Node
from rclpy.executors import ExternalShutdownException
from std_msgs.msg import Float64MultiArray, Float32MultiArray, UInt16MultiArray, Int16MultiArray, Int8, String, UInt16, Int8MultiArray, Bool
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Joy,Imu, Image
from std_srvs.srv import SetBool, Trigger

import os
import numpy as np
import time
from itertools import product
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import requests  # Used to send data to the plot server
import json      # Used to format the data
import threading
from ament_index_python.packages import get_package_share_directory

from include.cpg_rbf.cpg_so2 import CPG_SO2
from include.cpg_rbf.rbf import RBF
from include.cpg_rbf.cpg_so2 import CPG_LOCO
from include.cpg_rbf.dual_learner import DIL
from include.lowpass_filter import LPF
from include.gait_cycle_cut.gait_cycle_cut import OnlineGaitSegmenter
from include.enviroment_classification.env_pred import TimeSeriesKMeans


NAME_EXP = "dataset/air_rough_1"

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

        self.imitated_weights = np.load('imitated_diving_beetle_walk_forward_weights.npz') # load emtpy weights for initialization

        # swimming pattern weights
        # self.imitated_weights_swimming = np.load('imitated_diving_beetle_swim_forward_weights.npz')
        # self.imitated_weights_swimming = np.load('imitated_diving_beetle_swim_backward_weights.npz')
        # self.imitated_weights_swimming = np.load('imitated_diving_beetle_swim_turning_weights.npz')
        self.imitated_weights_swimming = np.load('imitated_diving_beetle_swim_crossing_weights.npz')

        # walking pattern weights
        self.imitated_weights_walking = np.load('imitated_diving_beetle_walk_forward_weights.npz')


        # ROS2 publishers and subscribers
        self.pub_jcmd                   = self.create_publisher(Float32MultiArray, '/diving_beetle/joint_angle_commands', 1)
        self.pub_expected_force         = self.create_publisher(Float32MultiArray, '/diving_beetle/expected_foot_force', 1)
        self.pub_foot_force             = self.create_publisher(Float32MultiArray, '/diving_beetle/resultant_foot_force_feedback', 1)
        self.pub_foot_force_error       = self.create_publisher(Float32MultiArray, '/diving_beetle/foot_force_error', 1)

        self.pub_cpg_output_0             = self.create_publisher(Float32MultiArray, '/diving_beetle/cpg_output_0', 1)
        self.pub_cpg_output_1             = self.create_publisher(Float32MultiArray, '/diving_beetle/cpg_output_1', 1)
        # self.pub_phi_cmd            = self.create_publisher(Float32MultiArray, '/diving_beetle/phi_cmd', 1)
        # self.pub_phi_walk           = self.create_publisher(Float32MultiArray, '/diving_beetle/phi_walk', 1)

        self.pub_env_class              = self.create_publisher(UInt16, '/diving_beetle/environment_class', 1)
        
        self.sub_js                     = self.create_subscription(Joy, '/joy', self.Joy_set, 1)        
        self.sub_terminated_cmd         = self.create_subscription(Bool, '/diving_beetle/terminated_cmd', self.TerminatedCmdCallback, 1)
        self.sub_started_cmd            = self.create_subscription(Bool, '/diving_beetle/started_cmd', self.StartedCmdCallback, 1)
        self.sub_execute_control_cmd    = self.create_subscription(Bool, '/diving_beetle/execute_control_cmd', self.ExecuteControlCmdCallback, 1)
        
        self.sub_foot_force             = self.create_subscription(Float32MultiArray, '/diving_beetle/foot_force_feedback', self.FootForceFeedback, 1)
        self.sub_joint_angles           = self.create_subscription(Float32MultiArray, '/diving_beetle/joint_angle_fb', self.JointAngleFeedback, 1)
        
        self.sub_leg_stiffness          = self.create_subscription(Float32MultiArray, '/diving_beetle/leg_stiffness_fb', self.LegStiffnessFeedback, 1)
        self.sub_leg_damping            = self.create_subscription(Float32MultiArray, '/diving_beetle/leg_damping_fb', self.LegDampingFeedback, 1)
        self.sub_leg_torque_feedforward    = self.create_subscription(Float32MultiArray, '/diving_beetle/leg_torque_feedforward_fb', self.LegTorqueFeedforwardFeedback, 1)

        # Command variables
        self.initial_joint_angles = {'FR': [ 0,  0, 0, 0],
                                     'BR': [ 0,  0, 0, 0],
                                     'FL': [ 0,  0, 0, 0],
                                     'BL': [ 0,  0, 0, 0]}  # degree
        self.initial_joint_angles = np.deg2rad(self.conv2float_arr(self.initial_joint_angles))
        self.joint_angles_cmd = {'FR': [ 0.9, 0.0,  0.266 , -1.5],
                                 'BR': [ -0.9, 0.0,  0.266 , -1.5],
                                 'FL': [ -0.9, 0.0,  0.266 , -1.5],
                                 'BL': [ 0.9, 0.0,  0.266 , -1.5]} # radian

        # Feedback variables
        self.joint_angles_fb = self.joint_angles_cmd.copy()
        self.leg_stiffness_fb   = {'FR': 0.0, 'BR': 0.0, 'FL': 0.0, 'BL': 0.0}
        self.leg_damping_fb     = {'FR': 0.0, 'BR': 0.0, 'FL': 0.0, 'BL': 0.0}
        self.leg_torque_feedforward_fb = {'FR': 0.0, 'BR': 0.0, 'FL': 0.0, 'BL': 0.0}

        self.leg_stiffness_cycle_buffer = {'FR': [], 'BR': [], 'FL': [], 'BL': []}
        self.leg_damping_cycle_buffer = {'FR': [], 'BR': [], 'FL': [], 'BL': []}
        self.leg_torque_feedforward_cycle_buffer = {'FR': [], 'BR': [], 'FL': [], 'BL': []}

        # Control variables
        self.cpg_modulated = {}
        self.cpg_output = {}
        self.cpg_mod_cmd = {}
        self.online_segmenter = {}
        self.online_environment_classifier = {}
        for side in LEG_SIDE:
            for index in LEG_INDEX:
                self.cpg_modulated[f'{index}{side}']     = CPG_LOCO()
                self.cpg_output[f'{index}{side}']        = self.cpg_modulated[f'{index}{side}'].modulate_cpg(0.05, 0.0, 1.0)  # initial run to set up internal states
                self.cpg_mod_cmd[f'{index}{side}']       = {'phi':0.2, 'pause_input': 1.0, 'rewind_input': 1.0}   # initial run to set up internal states
                self.online_segmenter[f'{index}{side}']  = OnlineGaitSegmenter()
        self.online_environment_classifier['FR'] = TimeSeriesKMeans()
        self.online_environment_classifier['FR'].load_model("/home/runj/BRAIN/InteractiveSearchLocomotion/software/project/alpha/ros2_ws/src/diving_beetle_pkg/diving_beetle_pkg/include/enviroment_classification/environment_classifier_model_2.npz")

        # Robot mode variables
        self.prev_x_button_state = False
        self.x_button_state = False
        self.robot_mode         = 'START'  # 'IDLE', 'START', 'PAUSE'
        self.robot_mode_dict    = {
            0:'PAUSE',
            1:'START',
        }
        self.robot_mode_index   = 0

        # Pattern mode variables
        self.prev_y_button_state = False
        self.y_button_state = False
        self.pattern_mode         = 'WALK'
        self.pattern_mode_dict    = {
            0:'WALK',
            1:'SWIM',
        }
        self.pattern_mode_index   = 0
        
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

        ros_node_freq  = 50
        self.timer      = self.create_timer(1/ros_node_freq, self.timer_callback_2)
        
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
    def JointAngleFeedback(self, msg):
        ...
        self.joint_angles_fb['FR'] = msg.data[0:4]
        self.joint_angles_fb['BR'] = msg.data[4:8]
        self.joint_angles_fb['FL'] = msg.data[8:12]
        self.joint_angles_fb['BL'] = msg.data[12:16]

    def LegStiffnessFeedback(self, msg):
        ...
        self.leg_stiffness_fb['FR'] = msg.data[0]
        self.leg_stiffness_fb['BR'] = msg.data[1]
        self.leg_stiffness_fb['FL'] = msg.data[2]
        self.leg_stiffness_fb['BL'] = msg.data[3]
    def LegDampingFeedback(self, msg):
        ...
        self.leg_damping_fb['FR'] = msg.data[0]
        self.leg_damping_fb['BR'] = msg.data[1]
        self.leg_damping_fb['FL'] = msg.data[2]
        self.leg_damping_fb['BL'] = msg.data[3]
    def LegTorqueFeedforwardFeedback(self, msg):
        ...
        self.leg_torque_feedforward_fb['FR'] = msg.data[0]
        self.leg_torque_feedforward_fb['BR'] = msg.data[1]
        self.leg_torque_feedforward_fb['FL'] = msg.data[2]
        self.leg_torque_feedforward_fb['BL'] = msg.data[3]

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

            # self.plot_full_robot_data(self.leg_stiffness_cycle_buffer, self.leg_damping_cycle_buffer)

            # self.save_locomotion_data()  # Save data when termination command is received
            # print("Saved data to .npz files.")

            self.destroy_node()
            rclpy.shutdown()

    # ==============================================
    def conv2float_arr(self,  input_dict):
        # return [float(value) for value in input_dict.values()]
        return [float(angle) for angles in input_dict.values() for angle in angles]
    def save_locomotion_data(self):
        """Entry point to trigger the background save."""
        # Create a copy of the data so the buffer can be cleared immediately 
        # without affecting the saving process
        data_snapshot = {}
        for leg, buffer in self.leg_stiffness_cycle_buffer.items():
            data_snapshot[f'stiffness_{leg}'] = list(buffer)
        for leg, buffer in self.leg_damping_cycle_buffer.items():
            data_snapshot[f'damping_{leg}'] = list(buffer)
        for leg, buffer in self.leg_torque_feedforward_cycle_buffer.items():
            data_snapshot[f'torque_feedforward_{leg}'] = list(buffer)

        # Start the thread
        save_thread = threading.Thread(target=self._background_save_task, args=(data_snapshot,))
        save_thread.start()
    
        # Optional: Clear your buffers here so they are ready for the next trial
        # while the thread works on the snapshot
        for leg in ['FR', 'BR', 'FL', 'BL']:
            self.leg_stiffness_cycle_buffer[leg].clear()
            self.leg_damping_cycle_buffer[leg].clear()
    def _background_save_task(self, data_dict):
        """The actual heavy lifting done by the thread."""
        try:
            # Convert lists to numpy objects
            processed_data = {k: np.array(v, dtype=object) for k, v in data_dict.items()}
            
            filename = f"{NAME_EXP}.npz"
            # Using savez instead of savez_compressed is faster for threads
            np.savez_compressed(filename, **processed_data)
            
            # In ROS2, use the thread-safe logger to confirm
            print(f"Background save complete: {filename}")
        except Exception as e:
            print(f"Failed to save in background: {e}")            
    # ==============================================
    def ChangeRobotMode(self):
        ''' 
        change the robot mode 
        
        robot_mode_dict: the dictionary of the robot mode (can be add/remove the mode at this dict.)
        '''

        self.robot_mode_index = (self.robot_mode_index + 1) % len(self.robot_mode_dict)
        self.robot_mode = self.robot_mode_dict[self.robot_mode_index]
        print(f"=========[STATUS] Switch Mode to: {self.robot_mode}=========")
    
    def ChangePatternMode(self):
        ''' 
        change the pattern mode 
        
        pattern_mode_dict: the dictionary of the pattern mode (can be add/remove the mode at this dict.)
        '''

        self.pattern_mode_index = (self.pattern_mode_index + 1) % len(self.pattern_mode_dict)
        self.pattern_mode = self.pattern_mode_dict[self.pattern_mode_index]
        print(f"=========[STATUS] Switch Mode to: {self.pattern_mode}=========")

    def timer_callback(self):
        if self.start_flag:
            self.control_loop()
    def timer_callback_2(self):
        self.capture_gait_cycle_data_loop()
    def control_loop(self):

        # ========= [Change Robot Mode] =========
        self.x_button_state = bool(self.btn_js[2])    # X button 'change robot mode'
        if self.prev_x_button_state == False and self.x_button_state == True: 
            self.ChangeRobotMode()
        self.prev_x_button_state = self.x_button_state

        # ========= [Change Robot Pattern] =========
        self.y_button_state = bool(self.btn_js[3])    # Y button 'change robot pattern'
        if self.prev_y_button_state == False and self.y_button_state == True: 
            self.ChangePatternMode()
        self.prev_y_button_state = self.y_button_state

        # ==================== Controller layer ====================
        if self.robot_mode == 'START':
            for side in LEG_SIDE:
                for index in LEG_INDEX:
                    self.cpg_mod_cmd[f'{index}{side}']['pause_input'] = 0.0
        elif self.robot_mode == 'PAUSE':
            for side in LEG_SIDE:
                for index in LEG_INDEX:
                    self.cpg_mod_cmd[f'{index}{side}']['pause_input'] = 1.0
        


        if self.pattern_mode == 'WALK':
            self.imitated_weights = self.imitated_weights_walking
            for side in LEG_SIDE:
                for index in LEG_INDEX:
                    self.cpg_mod_cmd[f'{index}{side}']['phi'] = 0.07
            
        elif self.pattern_mode == 'SWIM':
            self.imitated_weights = self.imitated_weights_swimming
            for side in LEG_SIDE:
                for index in LEG_INDEX:
                    self.cpg_mod_cmd[f'{index}{side}']['phi'] = 0.1
        
        # ==================== CPF modulation layer ====================  
        if self.robot_mode != 'IDLE':
            for side in LEG_SIDE:
                for index in LEG_INDEX:
                    self.cpg_output[f'{index}{side}'] = self.cpg_modulated[f'{index}{side}'].modulate_cpg(self.cpg_mod_cmd[f'{index}{side}']['phi'], 
                                                                                                        self.cpg_mod_cmd[f'{index}{side}']['pause_input'], 
                                                                                                        self.cpg_mod_cmd[f'{index}{side}']['rewind_input'])
            # ==================== RBF layer ====================
            for side in LEG_SIDE:
                for joint in JOINT_NAMES:
                    for index in LEG_INDEX:
                        self.joint_angles_cmd[f'{index}{side}'][joint] = self.rbf.regenerate_target_traj(self.cpg_output[f'{index}{side}']['cpg_output_0'], 
                                                                                                         self.cpg_output[f'{index}{side}']['cpg_output_1'],
                                                                                                         self.imitated_weights[f'{index}{side}{joint}'])
                        
        # if (cycle := self.online_segmenter['FR'].add_data_point(self.cpg_output['FR']['cpg_output_0'], [self.leg_stiffness_fb['FR'], self.leg_damping_fb['FR']])) is not None: self.leg_stiffness_cycle_buffer['FR'].append(cycle) self.leg_damping_cycle_buffer['FR'].append(cycle)
        # if (cycle := self.online_segmenter['BR'].add_data_point(self.cpg_output['BR']['cpg_output_0'], [self.leg_stiffness_fb['BR'], self.leg_damping_fb['BR']])) is not None: self.leg_stiffness_cycle_buffer['BR'].append(cycle) self.leg_damping_cycle_buffer['BR'].append(cycle)
        # if (cycle := self.online_segmenter['FL'].add_data_point(self.cpg_output['FL']['cpg_output_0'], [self.leg_stiffness_fb['FL'], self.leg_damping_fb['FL']])) is not None: self.leg_stiffness_cycle_buffer['FL'].append(cycle) self.leg_damping_cycle_buffer['FL'].append(cycle)
        # if (cycle := self.online_segmenter['BL'].add_data_point(self.cpg_output['BL']['cpg_output_0'], [self.leg_stiffness_fb['BL'], self.leg_damping_fb['BL']])) is not None: self.leg_stiffness_cycle_buffer['BL'].append(cycle) self.leg_damping_cycle_buffer['BL'].append(cycle)
        

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


            msg_float32.data = [self.cpg_output['FR']['cpg_output_0'], self.cpg_output['BR']['cpg_output_0'], self.cpg_output['FL']['cpg_output_0'], self.cpg_output['BL']['cpg_output_0']]
            self.pub_cpg_output_0.publish(msg_float32)
            msg_float32.data = [self.cpg_output['FR']['cpg_output_1'], self.cpg_output['BR']['cpg_output_1'], self.cpg_output['FL']['cpg_output_1'], self.cpg_output['BL']['cpg_output_1']]
            self.pub_cpg_output_1.publish(msg_float32)


    def capture_gait_cycle_data_loop(self):
        for leg in ['FR', 'BR', 'FL', 'BL']:
            # 1. Get the data point (passing stiffness and damping together)
            # The segmenter now returns a list of [stiffness, damping] pairs for the whole cycle
            if (cycle := self.online_segmenter[leg].add_data_point(
                self.cpg_output[leg]['cpg_output_0'], 
                [self.leg_stiffness_fb[leg], 
                 self.leg_damping_fb[leg],
                 self.leg_torque_feedforward_fb[leg]] 
            )) is not None:
                
                # 2. Convert to numpy array for easy slicing
                cycle_np = np.array(cycle) 
                
                # 3. Split and append (Slicing: [all_rows, column_index])
                self.leg_stiffness_cycle_buffer[leg].append(cycle_np[:, 0].tolist())
                self.leg_damping_cycle_buffer[leg].append(cycle_np[:, 1].tolist())
                self.leg_torque_feedforward_cycle_buffer[leg].append(cycle_np[:, 2].tolist())
                
                
                # if leg == 'FR':
                #     cycle = np.stack([cycle_np[:, 0].tolist(), 
                #              cycle_np[:, 1].tolist(),
                #              cycle_np[:, 2].tolist()], axis=1)
                #     cycle = self.online_environment_classifier['FR'].resample_cycle(cycle, target_len=90)
                #     cluster_id, uncertainty  = self.online_environment_classifier['FR'].predict(cycle)
                #     print(f"Environment Classification - Environment: {'ground' if cluster_id == 0 else 'water'}, Uncertainty: {uncertainty:.4f}")
                #     self.pub_env_class.publish(UInt16(data=int(cluster_id)))

    def plot_full_robot_data(self, stiffness_buffers, damping_buffers):
        legs = ['FR', 'FL', 'BR', 'BL']
        fig, axes = plt.subplots(4, 3, figsize=(12, 16), sharex=True)
        
        for i, leg in enumerate(legs):
            # Plot Stiffness
            for cycle in stiffness_buffers[leg]:
                axes[i, 0].plot(np.linspace(0, 100, len(cycle)), cycle, alpha=0.4, color='red')
            axes[i, 0].set_ylabel(f'{leg} Stiffness')
            
            # Plot Damping
            for cycle in damping_buffers[leg]:
                axes[i, 1].plot(np.linspace(0, 100, len(cycle)), cycle, alpha=0.4, color='orange')
            axes[i, 1].set_ylabel(f'{leg} Damping')

            # Plot Torque Feedforward
            for cycle in self.leg_torque_feedforward_cycle_buffer[leg]:
                axes[i, 2].plot(np.linspace(0, 100, len(cycle)), cycle, alpha=0.4, color='green')
            axes[i, 2].set_ylabel(f'{leg} Torque Feedforward')

        axes[3, 0].set_xlabel('Gait Cycle (%)')
        axes[3, 1].set_xlabel('Gait Cycle (%)')
        axes[3, 2].set_xlabel('Gait Cycle (%)')
        plt.tight_layout()
        plt.show()



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


