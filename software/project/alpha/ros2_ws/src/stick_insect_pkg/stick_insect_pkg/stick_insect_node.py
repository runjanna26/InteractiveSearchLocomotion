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
from include.cpg_rbf.dual_learner import DIL
from include.lowpass_filter import LPF


class StickInsectNode(Node):
    def __init__(self):
        super().__init__('stick_insect_node')
        
        self.rbf = RBF()
        self.cpg = CPG_SO2()

        # It would be centerize the kernel first
        cpg_one_cycle = self.cpg.generate_cpg_one_cycle(0.05)    # lowest phi value
        out0_cpg_one_cycle = cpg_one_cycle['out0_cpg_one_cycle'][:] # Use full cycle of CPG
        out1_cpg_one_cycle = cpg_one_cycle['out1_cpg_one_cycle'][:] # Use full cycle of CPG
        cpg_cycle_length = len(out0_cpg_one_cycle)
        self.rbf.construct_kernels_with_cpg_one_cycle(out0_cpg_one_cycle, out1_cpg_one_cycle, cpg_cycle_length) # construct kernels with cpg one cycle


        self.imitated_weights = np.load('imitated_stick_insect_right_leg_weights_me.npz')
        self.imitated_forces_weights = np.load('imitated_stick_insect_right_leg_weights_me_force.npz')
        # print(np.load('imitated_stick_insect_right_leg_weights_me_force.npz').files)

        # ROS2 publishers and subscribers
        self.pub_jcmd               = self.create_publisher(Float32MultiArray, '/stick_insect/joint_angle_commands', 1)
        self.pub_expected_force     = self.create_publisher(Float32MultiArray, '/stick_insect/expected_foot_force', 1)
        self.pub_foot_force         = self.create_publisher(Float32MultiArray, '/stick_insect/resultant_foot_force_feedback', 1)
        self.pub_foot_force_error   = self.create_publisher(Float32MultiArray, '/stick_insect/foot_force_error', 1)
        self.pub_phi_cmd            = self.create_publisher(Float32MultiArray, '/stick_insect/phi_cmd', 1)
        self.pub_phi_walk           = self.create_publisher(Float32MultiArray, '/stick_insect/phi_walk', 1)
        
        self.sub_js                 = self.create_subscription(Joy, '/joy', self.Joy_set, 1)
        self.sub_foot_force         = self.create_subscription(Float32MultiArray, '/stick_insect/foot_force_feedback', self.FootForceFeedback, 1)

        # Command variables
        self.initial_joint_angles = {'TR': [30,    0, -40],
                                     'CR': [10,    0,  10],
                                     'FR': [-60, -60, -60],
                                     'TL': [ 30,   0, -40],
                                     'CL': [ 10,   0,  10],
                                     'FL': [-60, -60, -60]}
        self.initial_joint_angles = np.deg2rad(self.conv2float_arr(self.initial_joint_angles))
        

        self.joint_angles_cmd = {'TR': [0.0, 0.0 ,0.0],
                                 'CR': [0.0, 0.0 ,0.0],
                                 'FR': [0.0, 0.0 ,0.0],
                                 'TL': [0.0, 0.0 ,0.0],
                                 'CL': [0.0, 0.0 ,0.0],
                                 'FL': [0.0, 0.0 ,0.0]}
        self.expected_foot_forces = {'R_Z0': 0.0,
                                     'R_Z1': 0.0,
                                     'R_Z2': 0.0,
                                     'L_Z0': 0.0,
                                     'L_Z1': 0.0,
                                     'L_Z2': 0.0}
        # Feedback variables
        self.foot_force_fb = {'F_R0': 0.0,
                              'F_R1': 0.0,
                              'F_R2': 0.0,
                              'F_L0': 0.0,
                              'F_L1': 0.0,
                              'F_L2': 0.0}
        self.foot_force_error = self.expected_foot_forces.copy() # only z-axis
        self.foot_force_tau   = self.expected_foot_forces.copy() # only z-axis
        
        self.axes_js = np.zeros(8)
        self.btn_js = np.zeros(8)

        self.start_time = time.time()  # Record the start time
        ros_node_freq = 50 #Hz
        self.timer = self.create_timer((1/ros_node_freq), self.timer_callback)
        
        
        # Control variables
        self.cpg_loco_leg_R0 = CPG_LOCO()
        self.cpg_loco_leg_R1 = CPG_LOCO()
        self.cpg_loco_leg_R2 = CPG_LOCO()
        self.cpg_loco_leg_L0 = CPG_LOCO()
        self.cpg_loco_leg_L1 = CPG_LOCO()
        self.cpg_loco_leg_L2 = CPG_LOCO()

        
        
        # First Params: work!
        # self.dual_learner_leg_R0 = DIL(0.54, 0.004, 0.0, 0.998, 0.0002, 0.0)
        # self.dual_learner_leg_R1 = DIL(0.54, 0.004, 0.0, 0.998, 0.0002, 0.0)
        # self.dual_learner_leg_R2 = DIL(0.54, 0.004, 0.0, 0.998, 0.0002, 0.0)
        # self.dual_learner_leg_L0 = DIL(0.54, 0.004, 0.0, 0.998, 0.0002, 0.0)
        # self.dual_learner_leg_L1 = DIL(0.54, 0.004, 0.0, 0.998, 0.0002, 0.0)
        # self.dual_learner_leg_L2 = DIL(0.54, 0.004, 0.0, 0.998, 0.0002, 0.0)
        
        self.dual_learner_leg_R0 = DIL(0.54, 0.006, 0.0, 0.998, 0.0002, 0.0)
        self.dual_learner_leg_R1 = DIL(0.54, 0.006, 0.0, 0.998, 0.0002, 0.0)
        self.dual_learner_leg_R2 = DIL(0.54, 0.006, 0.0, 0.998, 0.0002, 0.0)
        self.dual_learner_leg_L0 = DIL(0.54, 0.006, 0.0, 0.998, 0.0002, 0.0)
        self.dual_learner_leg_L1 = DIL(0.54, 0.006, 0.0, 0.998, 0.0002, 0.0)
        self.dual_learner_leg_L2 = DIL(0.54, 0.006, 0.0, 0.998, 0.0002, 0.0)
        
        self.cpg_output_leg_R0 = self.cpg_loco_leg_R0.modulate_cpg(0.05, 0.0, 1.0)
        self.cpg_output_leg_R1 = self.cpg_loco_leg_R1.modulate_cpg(0.05, 0.0, 1.0)
        self.cpg_output_leg_R2 = self.cpg_loco_leg_R2.modulate_cpg(0.05, 0.0, 1.0)
        self.cpg_output_leg_L0 = self.cpg_loco_leg_L0.modulate_cpg(0.05, 0.0, 1.0)
        self.cpg_output_leg_L1 = self.cpg_loco_leg_L1.modulate_cpg(0.05, 0.0, 1.0)
        self.cpg_output_leg_L2 = self.cpg_loco_leg_L2.modulate_cpg(0.05, 0.0, 1.0)
        
        self.cpg_mod_cmd = {'R0': {'phi':0.08, 'pause_input': 0.0, 'rewind_input': 1.0},
                            'R1': {'phi':0.08, 'pause_input': 0.0, 'rewind_input': 1.0},
                            'R2': {'phi':0.08, 'pause_input': 0.0, 'rewind_input': 1.0},
                            'L0': {'phi':0.08, 'pause_input': 0.0, 'rewind_input': 1.0},
                            'L1': {'phi':0.08, 'pause_input': 0.0, 'rewind_input': 1.0},
                            'L2': {'phi':0.08, 'pause_input': 0.0, 'rewind_input': 1.0},}
        self.cpg_phi_leg_des = {'R0': 0.05,
                                'R1': 0.05,
                                'R2': 0.05,
                                'L0': 0.05,
                                'L1': 0.05,
                                'L2': 0.05}
        self.cpg_phi_leg_adapt = {'R0': 0.0,
                                  'R1': 0.0,
                                  'R2': 0.0,
                                  'L0': 0.0,
                                  'L1': 0.0,
                                  'L2': 0.0}
        
        self.cpg_phi_cmd = 0.0
        self.cpg_pause_cmd = 0.0
        self.cpg_rewind_cmd = 1.0
        
        self.lpf_R0 = LPF(0.9)
        self.lpf_R1 = LPF(0.9)
        self.lpf_R2 = LPF(0.9)
        self.lpf_L0 = LPF(0.9)
        self.lpf_L1 = LPF(0.9)
        self.lpf_L2 = LPF(0.9)
        
        self.sig_1 = 0.0
        self.sig_2 = 0.0
        self.sig_1_list = []
        self.sig_2_list = []
        self.tau_1 = None
        self.tau_2 = None
        self.tau = 0.0
        self.start_flag = False
        self.end_flag = False
        
    # =============== Joy Stick ====================
    def Joy_set(self, msg):
        self.axes_js = msg.axes
        self.btn_js = msg.buttons  
        # print(self.axes_js)
        # print(self.btn_js)
    def FootForceFeedback(self, msg):
        scaling_factor = 10
        offset_factor = 0.15
        self.foot_force_fb['F_R0'] = np.clip(self.lpf_R0.filter(scaling_factor * msg.data[0]) - offset_factor, 0.0, 1.0)  # x, y, z
        self.foot_force_fb['F_R1'] = np.clip(self.lpf_R1.filter(scaling_factor * msg.data[1]) - offset_factor, 0.0, 1.0)
        self.foot_force_fb['F_R2'] = np.clip(self.lpf_R2.filter(scaling_factor * msg.data[2]) - offset_factor, 0.0, 1.0)
        self.foot_force_fb['F_L0'] = np.clip(self.lpf_L0.filter(scaling_factor * msg.data[3]) - offset_factor, 0.0, 1.0)
        self.foot_force_fb['F_L1'] = np.clip(self.lpf_L1.filter(scaling_factor * msg.data[4]) - offset_factor, 0.0, 1.0)
        self.foot_force_fb['F_L2'] = np.clip(self.lpf_L2.filter(scaling_factor * msg.data[5]) - offset_factor, 0.0, 1.0)
        # print(self.foot_force_fb)
        
    # ==============================================
    def conv2float_arr(self,  input_dict):
        # return [float(value) for value in input_dict.values()]
        return [float(angle) for angles in input_dict.values() for angle in angles]
    
    # def cross_correlation(self, in_1, in_2):
    #     current_time = time.time()  # Get current timestamp in seconds
        
    #     th = 0.2
    #     if in_1 >= th and self.in_1_prev < th:  # signal cross the threshold
    #         self.start_flag = True
    #     elif in_1 <= th and self.in_1_prev > th:
        
        
    #     self.sig_1_list.append(in_1)
    #     self.sig_2_list.append(in_2)       
         
    #     if in_2 >= th and self.in_2_prev < th:  # signal cross the threshold
    #         # self.tau_2 = current_time
    #         # print(self.tau_2)
        
    #     if in_1 >= th and in_2 >= th:
    #         print(self.tau_1 - self.tau_2)
            
    #     self.in_1_prev = in_1
    #     self.in_2_prev = in_2
            
    def get_phase_shift_time(self, in_1, in_2):
        current_time = time.time()  # Get current timestamp in seconds
        self.threshold = 0.15
        # Detect when in_1 crosses the threshold
        if in_1 >= self.threshold:
            self.tau_1 = current_time

        # Detect when in_2 crosses the threshold
        if in_2 >= self.threshold:
            self.tau_2 = current_time

        # Compute time delay when both timestamps are available
        if self.tau_1 is not None and self.tau_2 is not None:
            
            self.tau = self.tau_2 - self.tau_1  # Compute time delay (+ lead, - lag)
            
            if np.abs(self.tau) > 0.0:
                ...
                # print(f"Phase Shift (Tau): {self.tau:.6f} s")
            
            # Reset timestamps for the next cycle
            self.tau_1 = None
            self.tau_2 = None
        return np.sign(self.tau)
        
        

    # ==============================================
    def timer_callback(self):
        ...
        
        # Manual shift phase
        if self.btn_js[0] == 1: # A button
            self.cpg_mod_cmd['R0']['phi'] = 0.01
            self.cpg_mod_cmd['R1']['phi'] = 0.05
            self.cpg_mod_cmd['R2']['phi'] = 0.01

            self.cpg_mod_cmd['L0']['phi'] = 0.05
            self.cpg_mod_cmd['L1']['phi'] = 0.01
            self.cpg_mod_cmd['L2']['phi'] = 0.05
        else:
            self.cpg_mod_cmd['R0']['phi'] = 0.05
            self.cpg_mod_cmd['R1']['phi'] = 0.05
            self.cpg_mod_cmd['R2']['phi'] = 0.05

            self.cpg_mod_cmd['L0']['phi'] = 0.05
            self.cpg_mod_cmd['L1']['phi'] = 0.05
            self.cpg_mod_cmd['L2']['phi'] = 0.05
            
        # if self.btn_js[0] == 1: # A button
        #     self.cpg_mod_cmd['R0']['pause_input'] = 1.0
        #     self.cpg_mod_cmd['R1']['pause_input'] = 0.0
        #     self.cpg_mod_cmd['R2']['pause_input'] = 1.0

        #     self.cpg_mod_cmd['L0']['pause_input'] = 0.0
        #     self.cpg_mod_cmd['L1']['pause_input'] = 1.0
        #     self.cpg_mod_cmd['L2']['pause_input'] = 0.0
        # else:
        #     self.cpg_mod_cmd['R0']['pause_input'] = 0.0
        #     self.cpg_mod_cmd['R1']['pause_input'] = 0.0
        #     self.cpg_mod_cmd['R2']['pause_input'] = 0.0

        #     self.cpg_mod_cmd['L0']['pause_input'] = 0.0
        #     self.cpg_mod_cmd['L1']['pause_input'] = 0.0
        #     self.cpg_mod_cmd['L2']['pause_input'] = 0.0
        # if self.btn_js[1] == 1: # B button
        #     self.cpg_phi_cmd = 0.1
        #     self.cpg_pause_cmd = 0.0
        #     self.cpg_rewind_cmd = -1.0
        # if self.btn_js[2] == 1: # X button
        #     self.cpg_phi_cmd = 0.1
        #     self.cpg_pause_cmd = 1.0
        
        # ==================== Self-orgnization for walking pattern ====================
        
        # for col_fb, col_expec in zip(self.foot_force_fb, self.expected_foot_forces):
        #     # print(col_fb, col_expec)
        #     self.foot_force_tau[col_expec]   = self.get_phase_shift_time(self.expected_foot_forces[col_expec], self.foot_force_fb[col_fb])
            
        #     # if self.foot_force_tau[col_expec] is not None:
        #     # print()
            
        #     self.foot_force_error[col_expec] =  self.foot_force_tau[col_expec] * np.abs(self.foot_force_fb[col_fb] - self.expected_foot_forces[col_expec])
        # self.pub_foot_force.publish(Float32MultiArray(data = list(self.foot_force_fb.values())))
        # self.pub_foot_force_error.publish(Float32MultiArray(data = list(self.foot_force_error.values())))
        
        
        
        # self.cpg_phi_leg_adapt['R_Z0'], _, _ = self.dual_learner_leg_R0.calculate_DIL(self.foot_force_error['R_Z0'])
        # self.cpg_phi_leg_adapt['R_Z1'], _, _ = self.dual_learner_leg_R1.calculate_DIL(self.foot_force_error['R_Z1'])
        # self.cpg_phi_leg_adapt['R_Z2'], _, _ = self.dual_learner_leg_R2.calculate_DIL(self.foot_force_error['R_Z2'])
        # self.cpg_phi_leg_adapt['L_Z0'], _, _ = self.dual_learner_leg_L0.calculate_DIL(self.foot_force_error['L_Z0'])
        # self.cpg_phi_leg_adapt['L_Z1'], _, _ = self.dual_learner_leg_L1.calculate_DIL(self.foot_force_error['L_Z1'])
        # self.cpg_phi_leg_adapt['L_Z2'], _, _ = self.dual_learner_leg_L2.calculate_DIL(self.foot_force_error['L_Z2'])
        
        # self.cpg_mod_cmd['R0']['phi'] = self.cpg_phi_leg_adapt['R_Z0'] + self.cpg_phi_leg_des['R0']
        # self.cpg_mod_cmd['R1']['phi'] = self.cpg_phi_leg_adapt['R_Z1'] + self.cpg_phi_leg_des['R1']
        # self.cpg_mod_cmd['R2']['phi'] = self.cpg_phi_leg_adapt['R_Z2'] + self.cpg_phi_leg_des['R2']
        # self.cpg_mod_cmd['L0']['phi'] = self.cpg_phi_leg_adapt['L_Z0'] + self.cpg_phi_leg_des['L0']
        # self.cpg_mod_cmd['L1']['phi'] = self.cpg_phi_leg_adapt['L_Z1'] + self.cpg_phi_leg_des['L1']
        # self.cpg_mod_cmd['L2']['phi'] = self.cpg_phi_leg_adapt['L_Z2'] + self.cpg_phi_leg_des['L2']
        # self.pub_phi_cmd.publish(Float32MultiArray(data = [self.cpg_mod_cmd['R0']['phi'], 
        #                                                    self.cpg_mod_cmd['R1']['phi'], 
        #                                                    self.cpg_mod_cmd['R2']['phi'], 
        #                                                    self.cpg_mod_cmd['L0']['phi'], 
        #                                                    self.cpg_mod_cmd['L1']['phi'], 
        #                                                    self.cpg_mod_cmd['L2']['phi']]))
        
        
        # self.pub_phi_walk.publish(Float32MultiArray(data = [self.cpg_phi_leg_des ['R0'],
        #                                                     self.cpg_phi_leg_des ['R1'],
        #                                                     self.cpg_phi_leg_des ['R2'],
        #                                                     self.cpg_phi_leg_des ['L0'],
        #                                                     self.cpg_phi_leg_des ['L1'],
        #                                                     self.cpg_phi_leg_des ['L2']]))
        
               
        # ==================== CPF modulation layer ====================   
        self.cpg_output_leg_R0 = self.cpg_loco_leg_R0.modulate_cpg(self.cpg_mod_cmd['R0']['phi'] , self.cpg_mod_cmd['R0']['pause_input'], self.cpg_mod_cmd['R0']['rewind_input'])
        self.cpg_output_leg_R1 = self.cpg_loco_leg_R1.modulate_cpg(self.cpg_mod_cmd['R1']['phi'] , self.cpg_mod_cmd['R1']['pause_input'], self.cpg_mod_cmd['R1']['rewind_input'])
        self.cpg_output_leg_R2 = self.cpg_loco_leg_R2.modulate_cpg(self.cpg_mod_cmd['R2']['phi'] , self.cpg_mod_cmd['R2']['pause_input'], self.cpg_mod_cmd['R2']['rewind_input'])
        self.cpg_output_leg_L0 = self.cpg_loco_leg_L0.modulate_cpg(self.cpg_mod_cmd['L0']['phi'] , self.cpg_mod_cmd['L0']['pause_input'], self.cpg_mod_cmd['L0']['rewind_input'])
        self.cpg_output_leg_L1 = self.cpg_loco_leg_L1.modulate_cpg(self.cpg_mod_cmd['L1']['phi'] , self.cpg_mod_cmd['L1']['pause_input'], self.cpg_mod_cmd['L1']['rewind_input'])
        self.cpg_output_leg_L2 = self.cpg_loco_leg_L2.modulate_cpg(self.cpg_mod_cmd['L2']['phi'] , self.cpg_mod_cmd['L2']['pause_input'], self.cpg_mod_cmd['L2']['rewind_input'])
        
        
        # ==================== RBF layer ====================
        # Right legs
        self.joint_angles_cmd['TR'][0] = self.rbf.regenerate_target_traj(self.cpg_output_leg_R0['cpg_output_0'], self.cpg_output_leg_R0['cpg_output_1'], self.imitated_weights['TR0'])         
        self.joint_angles_cmd['CR'][0] = self.rbf.regenerate_target_traj(self.cpg_output_leg_R0['cpg_output_0'], self.cpg_output_leg_R0['cpg_output_1'], self.imitated_weights['CR0'])
        self.joint_angles_cmd['FR'][0] = self.rbf.regenerate_target_traj(self.cpg_output_leg_R0['cpg_output_0'], self.cpg_output_leg_R0['cpg_output_1'], self.imitated_weights['FR0'])
        self.expected_foot_forces['R_Z0'] = self.rbf.regenerate_target_traj(self.cpg_output_leg_R0['cpg_output_0'], self.cpg_output_leg_R0['cpg_output_1'], self.imitated_forces_weights['F0'])
        
        self.joint_angles_cmd['TR'][1] = self.rbf.regenerate_target_traj(self.cpg_output_leg_R1['cpg_output_0'], self.cpg_output_leg_R1['cpg_output_1'], self.imitated_weights['TR1'])         
        self.joint_angles_cmd['CR'][1] = self.rbf.regenerate_target_traj(self.cpg_output_leg_R1['cpg_output_0'], self.cpg_output_leg_R1['cpg_output_1'], self.imitated_weights['CR1'])
        self.joint_angles_cmd['FR'][1] = self.rbf.regenerate_target_traj(self.cpg_output_leg_R1['cpg_output_0'], self.cpg_output_leg_R1['cpg_output_1'], self.imitated_weights['FR1'])
        self.expected_foot_forces['R_Z1'] = self.rbf.regenerate_target_traj(self.cpg_output_leg_R1['cpg_output_0'], self.cpg_output_leg_R1['cpg_output_1'], self.imitated_forces_weights['F1'])
        
        self.joint_angles_cmd['TR'][2] = self.rbf.regenerate_target_traj(self.cpg_output_leg_R2['cpg_output_0'], self.cpg_output_leg_R2['cpg_output_1'], self.imitated_weights['TR2'])         
        self.joint_angles_cmd['CR'][2] = self.rbf.regenerate_target_traj(self.cpg_output_leg_R2['cpg_output_0'], self.cpg_output_leg_R2['cpg_output_1'], self.imitated_weights['CR2'])
        self.joint_angles_cmd['FR'][2] = self.rbf.regenerate_target_traj(self.cpg_output_leg_R2['cpg_output_0'], self.cpg_output_leg_R2['cpg_output_1'], self.imitated_weights['FR2'])
        self.expected_foot_forces['R_Z2'] = self.rbf.regenerate_target_traj(self.cpg_output_leg_R2['cpg_output_0'], self.cpg_output_leg_R2['cpg_output_1'], self.imitated_forces_weights['F2'])
        
        # Left legs use the weights of right legs
        self.joint_angles_cmd['TL'][0] = self.rbf.regenerate_target_traj(self.cpg_output_leg_L0['cpg_output_0'], self.cpg_output_leg_L0['cpg_output_1'], self.imitated_weights['TR0'])         
        self.joint_angles_cmd['CL'][0] = self.rbf.regenerate_target_traj(self.cpg_output_leg_L0['cpg_output_0'], self.cpg_output_leg_L0['cpg_output_1'], self.imitated_weights['CR0'])
        self.joint_angles_cmd['FL'][0] = self.rbf.regenerate_target_traj(self.cpg_output_leg_L0['cpg_output_0'], self.cpg_output_leg_L0['cpg_output_1'], self.imitated_weights['FR0'])
        self.expected_foot_forces['L_Z0'] = self.rbf.regenerate_target_traj(self.cpg_output_leg_L0['cpg_output_0'], self.cpg_output_leg_L0['cpg_output_1'], self.imitated_forces_weights['F0'])
        
        self.joint_angles_cmd['TL'][1] = self.rbf.regenerate_target_traj(self.cpg_output_leg_L1['cpg_output_0'], self.cpg_output_leg_L1['cpg_output_1'], self.imitated_weights['TR1'])         
        self.joint_angles_cmd['CL'][1] = self.rbf.regenerate_target_traj(self.cpg_output_leg_L1['cpg_output_0'], self.cpg_output_leg_L1['cpg_output_1'], self.imitated_weights['CR1'])
        self.joint_angles_cmd['FL'][1] = self.rbf.regenerate_target_traj(self.cpg_output_leg_L1['cpg_output_0'], self.cpg_output_leg_L1['cpg_output_1'], self.imitated_weights['FR1'])
        self.expected_foot_forces['L_Z1'] = self.rbf.regenerate_target_traj(self.cpg_output_leg_L1['cpg_output_0'], self.cpg_output_leg_L1['cpg_output_1'], self.imitated_forces_weights['F1'])
        
        self.joint_angles_cmd['TL'][2] = self.rbf.regenerate_target_traj(self.cpg_output_leg_L2['cpg_output_0'], self.cpg_output_leg_L2['cpg_output_1'], self.imitated_weights['TR2'])         
        self.joint_angles_cmd['CL'][2] = self.rbf.regenerate_target_traj(self.cpg_output_leg_L2['cpg_output_0'], self.cpg_output_leg_L2['cpg_output_1'], self.imitated_weights['CR2'])
        self.joint_angles_cmd['FL'][2] = self.rbf.regenerate_target_traj(self.cpg_output_leg_L2['cpg_output_0'], self.cpg_output_leg_L2['cpg_output_1'], self.imitated_weights['FR2'])
        self.expected_foot_forces['L_Z2'] = self.rbf.regenerate_target_traj(self.cpg_output_leg_L2['cpg_output_0'], self.cpg_output_leg_L2['cpg_output_1'], self.imitated_forces_weights['F2'])
        
        
        if time.time() - self.start_time < 2:
            # set initial posture
            msg_float32 = Float32MultiArray()
            msg_float32.data = [float(x) for x in self.initial_joint_angles]
            self.pub_jcmd.publish(msg_float32)
        elif time.time() - self.start_time >= 2:
            msg_float32 = Float32MultiArray()
            
            msg_float32.data = self.conv2float_arr(self.joint_angles_cmd)
            self.pub_jcmd.publish(msg_float32)
            
            msg_float32.data = list(self.expected_foot_forces.values())
            self.pub_expected_force.publish(msg_float32)


    

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


