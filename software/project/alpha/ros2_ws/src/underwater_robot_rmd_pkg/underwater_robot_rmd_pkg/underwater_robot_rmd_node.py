import time
import subprocess
import re
import paramiko
import threading
import numpy as np

import rclpy
from rclpy.node import Node, QoSProfile
from rclpy.qos import ReliabilityPolicy

from sensor_msgs.msg import Joy, CompressedImage, JointState
from std_msgs.msg import Float32MultiArray, UInt8MultiArray, String, Bool, Float32, UInt8

best_effort_qos_prof = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)


class RobotNode(Node):
    def __init__(self):
        super().__init__('underwater_robot_node')
        self.cmd_pub    = self.create_publisher(Float32MultiArray, '/command', 1)
        self.debug_pub  = self.create_publisher(Float32MultiArray, '/debugging', 1)

        # self.joint_states_sub = self.create_subscription(JointState, '/joint_states', self.joint_states_callback, best_effort_qos_prof)
        

        self.get_logger().info('Starting motor control...')

        ros_node_freq = 100 #Hz
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

        self.joint_position = np.zeros(1)
        self.joint_velocity = np.zeros(1)
        self.joint_torque   = np.zeros(1)

    # def joint_states_callback(self, msg:JointState):
    #     self.joint_position = msg.position
    #     self.joint_velocity = msg.velocity
    #     self.joint_torque   = msg.effort


    # =========================================================================================================
    #                                               ROS Loop
    # =========================================================================================================
    def timer_callback(self):
        
        current_time = self.get_clock().now().nanoseconds / 1e9  # Convert nanoseconds to seconds
        t = current_time - self.start_time  # Elapsed time since start


        # ======================= Start ROS Loop ========================= #
        # Setup sinusoidal oscillation parameters 
        A_max = 1.0                                     # Amplitude (radians)
        frequency = 1                                     # Frequency (Hz)
        ramp_time = 2                                       # Ramp time constant
        amplitude = A_max * (1 - np.exp(-t / ramp_time))    # Sine wave (radians)
        omega = 2 * np.pi * frequency    

        # pos_des = 0.0
        # vel_des = 0.0

        pos_des = amplitude * np.sin(omega * t)
        vel_des = amplitude * omega * np.cos(omega * t)




        # self.cmd_pub.publish(Float32MultiArray(data=[pos_des, vel_des, 0.09]))
        self.cmd_pub.publish(Float32MultiArray(data=[pos_des, vel_des, 
                                                     0.01, 0.0, -1.0]))

        # self.debug_pub.publish(Float32MultiArray(data=[self.joint_position[0],
        #                                                self.joint_velocity[0],
        #                                                self.joint_torque[0]]))


        # ======================= End ROS Loop ========================= #
        # Stop node after a period of time
        if t >= self.max_runtime and self.max_runtime != -1:
            self.get_logger().info(f"Remaining {self.iteration - 1} trial(s): {self.max_runtime} second(s)")
            self.start_time = self.get_clock().now().nanoseconds / 1e9  # Start time in seconds
            
            self.iteration -= 1
            if self.iteration <= 0:
                # self.can_manager.stop_all_motors()
                # self.can_manager.shutdown_all_motors()
                # self.can_manager.reset_all_motors()
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
        # node.can_manager.stop_all_motors()
        # node.can_manager.shutdown_all_motors()
        # node.can_manager.reset_all_motors()
        node.destroy_node()
        rclpy.shutdown()
        
if __name__ == '__main__':
    main()
