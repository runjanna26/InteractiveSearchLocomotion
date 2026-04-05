
import time
import subprocess
import re
import paramiko
import threading
import numpy as np

import rclpy
from rclpy.node import Node, QoSProfile
from rclpy.qos import ReliabilityPolicy

from sensor_msgs.msg import Joy, CompressedImage
from underwater_robot_pkg.as5047 import AS5X47
from std_msgs.msg import Float32MultiArray, UInt8MultiArray, String, Bool, Float32, UInt8

from underwater_robot_pkg.can_manager import CAN_Manager
from underwater_robot_pkg.rmd_motor_can import rmd_motor_can

from underwater_robot_pkg.lowpass_filter import LPF
from underwater_robot_pkg.muscle_model import MuscleModel
# from .muscle_model_new import MuscleModel


best_effort_qos_prof = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)

class RobotNode(Node):
    def __init__(self):
        super().__init__('underwater_robot_node')

        # Create publishers
        self.robot_joint_pos_des     = self.create_publisher(Float32MultiArray, '/joint_position_desired', best_effort_qos_prof)
        self.robot_joint_pos_fb      = self.create_publisher(Float32MultiArray, '/joint_position_feedback', best_effort_qos_prof)
        self.robot_joint_vel_des     = self.create_publisher(Float32MultiArray, '/joint_velocity_desired', best_effort_qos_prof)
        self.robot_joint_vel_fb      = self.create_publisher(Float32MultiArray, '/joint_velocity_feedback', best_effort_qos_prof)
        self.robot_joint_tor_fb      = self.create_publisher(Float32MultiArray, '/joint_torque_feedback', best_effort_qos_prof)

        self.robot_joint_k_des       = self.create_publisher(Float32MultiArray, '/joint_stiffness_desired', best_effort_qos_prof)
        self.robot_joint_d_des       = self.create_publisher(Float32MultiArray, '/joint_damping_desired', best_effort_qos_prof)
        self.robot_joint_tff_des     = self.create_publisher(Float32MultiArray, '/joint_feedforward_torque_desired', best_effort_qos_prof)

        self.robot_joint_cur_fb      = self.create_publisher(Float32MultiArray, '/joint_current_feedback', best_effort_qos_prof)
        self.robot_joint_vol_fb      = self.create_publisher(Float32MultiArray, '/joint_voltage_feedback', best_effort_qos_prof)

        self.robot_joint_tem_fb      = self.create_publisher(Float32MultiArray, '/joint_temperature_feedback', best_effort_qos_prof)
        self.robot_joint_mtem_fb     = self.create_publisher(Float32MultiArray, '/joint_mosfet_temperature_feedback', best_effort_qos_prof)

        self.robot_joint_err_fb      = self.create_publisher(String, '/joint_error_feedback', best_effort_qos_prof)

        self.joint_enc_pos_fb        = self.create_publisher(Float32MultiArray, '/joint_encoder_position_feedback', best_effort_qos_prof)

        self.debug_pub               = self.create_publisher(Float32MultiArray, 'debugging_output', best_effort_qos_prof)


        self.get_logger().info('Starting motor control...')

        self.can_manager = CAN_Manager()
        self.can_manager.check_all_motors()
        self.can_manager.reset_all_motors()
        
        self.motor_ids = [0x02, 0x01]  # List of motor IDs to initialize
        self.motors = []
        for motor_id in self.motor_ids:
            motor = rmd_motor_can(motor_id=motor_id, can_manager=self.can_manager)
            self.motors.append(motor)


    
        # self.encoder = AS5X47()

        self.LPF_vel = LPF(0.5)

        self.init_pos           = [- np.pi/2 - np.pi/4, - np.pi/4]
        # self.muscle = MuscleModel(_a            = 1.0,
        #                           _b            = 20.0,
        #                           _beta         = 0.0,             # made motor oscillation smaller after holding
        #                           _init_pos     = self.init_pos,
        #                           number_motor = 1)


        ros_node_freq = 1000 #Hz
        self.timer = self.create_timer((1/ros_node_freq), self.timer_callback)
        # self.max_runtime = 15  # seconds
        self.max_runtime = -1  # loop forever
        self.iteration = 1
        self.start_time = self.get_clock().now().nanoseconds / 1e9  # Start time in seconds
        self.get_logger().info(f'Node period {self.max_runtime} seconds, {self.iteration} iterations')

        
        # Initialize parameters
        self.axes_js                    = (0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
        self.btn_js                     = (0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
        self.prev_mode_button_state     = False
        self.mode_button_state          = False
        
        self.all_connection_msg         = [0,0,0,0,0,0,0,0]
        self.all_temperature_msg        = [0,0,0,0]

        self.position_enc               = 0.0
        
    # =========================================================================================================
    #                                               ROS Loop
    # =========================================================================================================
    def timer_callback(self):
        
        current_time = self.get_clock().now().nanoseconds / 1e9  # Convert nanoseconds to seconds
        t = current_time - self.start_time  # Elapsed time since start


        # ======================= Start ROS Loop ========================= #

        # Setup sinusoidal oscillation parameters 
        A_max = np.pi/4                                     # Amplitude (radians)
        frequency = 0.5                                       # Frequency (Hz)
        ramp_time = 2                                       # Ramp time constant
        amplitude = A_max * (1 - np.exp(-t / ramp_time))    # Sine wave (radians)
        omega = 2 * np.pi * frequency    

        pos_des = [amplitude * np.sin(omega * t), 
                   amplitude * np.sin(omega * t + np.pi/4)]  # Desired position (radians)
        vel_des = [amplitude * omega * np.cos(omega * t),
                   amplitude * omega * np.cos(omega * t + np.pi/4)]  # Desired velocity (radians/second)

        # pos_des = amplitude * np.sin(omega * t)
        # vel_des= amplitude * omega * np.cos(omega * t)
        


        # ========================= Online Adaptive Impedance Control ========================= #
        # self.muscle.calculate(pos_des = pos_des, 
        #                       pos_fb  = self.motor.feedback_multi_turn_position,        # It has been minus init_pos in class
        #                       vel_fb  = self.LPF_vel.filter(self.motor.feedback_velocity))               


        # ======================= Set motor command ======================= #

        self.motors[0].set_desired_position_radian(pos_des[0] + self.init_pos[0])  
        self.motors[0].set_desired_velocity_radian_per_second(vel_des[0])

        self.motors[0].set_desired_stiffness(20)
        self.motors[0].set_desired_damping(5)
        self.motors[0].set_desired_torque(0)
        
        self.motors[1].set_desired_position_radian(pos_des[1] + self.init_pos[1])
        self.motors[1].set_desired_velocity_radian_per_second(vel_des[1])

        self.motors[1].set_desired_stiffness(20)
        self.motors[1].set_desired_damping(5)
        self.motors[1].set_desired_torque(0)
        

        # self.motors[0].set_desired_stiffness(self.muscle.get_stiffness())
        # self.motors[0].set_desired_damping(self.muscle.get_damping())
        # self.motors[0].set_desired_torque(self.muscle.get_feedforward_force())

        # Direct torque control (No limit stiffness and damping)
        # self.motors[0].set_desired_torque(np.sign(self.motors[0].feedback_velocity)*0.2)  # Add friction compensation
        # self.motors[0].set_desired_torque(self.muscle.tau + np.sign(self.motors[0].feedback_velocity)*0.2) 
        # self.motors[0].set_desired_torque(self.muscle.tau) 


        # ======================= Send motor command ======================= #
        self.motors[0].update()
        self.motors[1].update()
        # ======================= Read feedbacks ======================= #
        # self.position_enc  = np.deg2rad(self.encoder.read_angle())
        # =========================== Publish messages =========================== #
        msg = Float32MultiArray()
        msg.data = [float(pos_des[0]), float(pos_des[1])]
        # msg.data = [self.motors[0].desired_position]
        self.robot_joint_pos_des.publish(msg)

        # msg.data = [self.muscle.vel_des]
        # self.robot_joint_vel_des.publish(msg)

        msg.data = [self.motors[0].feedback_multi_turn_position]
        self.robot_joint_pos_fb.publish(msg)

        msg.data = [self.motors[0].feedback_velocity]
        self.robot_joint_vel_fb.publish(msg)

        msg.data = [self.motors[0].feedback_motor_torque]
        self.robot_joint_tor_fb.publish(msg)

        msg.data = [self.motors[0].feedback_temperature]
        self.robot_joint_tem_fb.publish(msg)

        msg.data = [self.motors[0].feedback_mosfet_temperature]
        self.robot_joint_mtem_fb.publish(msg)

        msg.data = [self.motors[0].feedback_current]
        self.robot_joint_cur_fb.publish(msg)

        msg.data = [self.motors[0].feedback_voltage]
        self.robot_joint_vol_fb.publish(msg)

        # msg.data = [self.muscle.K]
        # self.robot_joint_k_des.publish(msg)  

        # msg.data = [self.muscle.D]
        # self.robot_joint_d_des.publish(msg)  

        # msg.data = [self.muscle.F]
        # self.robot_joint_tff_des.publish(msg)

        msg.data = [self.position_enc]
        self.joint_enc_pos_fb.publish(msg)


        if self.motors[0].error_state:
            self.get_logger().info(f'Error: {self.motors[0].errors}')

            str_msg = String()
            str_msg.data = ', '.join(self.motors[0].errors)
            self.robot_joint_err_fb.publish(str_msg)

        # Debugging publisher
        self.debug_pub.publish(Float32MultiArray(data=[]))

        # ======================= End ROS Loop ========================= #
        # Stop node after a period of time
        if t >= self.max_runtime and self.max_runtime != -1:
            self.get_logger().info(f"Remaining {self.iteration - 1} trial(s): {self.max_runtime} second(s)")
            self.start_time = self.get_clock().now().nanoseconds / 1e9  # Start time in seconds
            
            self.iteration -= 1
            if self.iteration <= 0:
                self.can_manager.stop_all_motors()
                self.can_manager.shutdown_all_motors()
                self.can_manager.reset_all_motors()
                self.destroy_node()
                rclpy.shutdown()
                return
       

# =========================================================================================================
#                                               Main
# =========================================================================================================
def main(args=None):
    rclpy.init(args=args)
    node = RobotNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("KeyboardInterrupt received, shutting down.")
    finally:
        # Cleanup ROS 2 node
        node.can_manager.stop_all_motors()
        node.can_manager.shutdown_all_motors()
        node.can_manager.reset_all_motors()
        node.destroy_node()
        rclpy.shutdown()
        
if __name__ == '__main__':
    main()
