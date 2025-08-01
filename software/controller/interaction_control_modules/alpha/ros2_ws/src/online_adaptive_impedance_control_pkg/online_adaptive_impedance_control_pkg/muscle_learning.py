import time
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.executors import ExternalShutdownException

from std_msgs.msg import Float64MultiArray, Float32MultiArray, UInt16MultiArray, Int16MultiArray, UInt8
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Joy,Imu

from mit_can import TMotorManager_mit_can
from lowpass_filter import LPF
from muscle_model import MuscleModel
# from muscle_model_new import MuscleModel

# from muscle_model_new_2 import MuscleModel

class MainNode(Node):
    def __init__(self):
        super().__init__('MuscleLearning_node')

        self.pub_q        = self.create_publisher(Float32MultiArray, "/JointAngles", 1)
        self.pub_q_dot    = self.create_publisher(Float32MultiArray, "/JointVelocities", 1)
        self.pub_qd       = self.create_publisher(Float32MultiArray, "/JointCommands", 1)
        self.pub_qd_dot   = self.create_publisher(Float32MultiArray, "/JointVelocityCommands", 1)
        self.pub_qe       = self.create_publisher(Float32MultiArray, "/JointAngleErrors", 1)

        self.pub_tor      = self.create_publisher(Float32MultiArray, "/JointTorques", 1)
        self.pub_k      = self.create_publisher(Float32MultiArray, "/JointStiffness", 1)
        self.pub_d      = self.create_publisher(Float32MultiArray, "/JointDamping", 1)
        self.pub_f      = self.create_publisher(Float32MultiArray, "/JointFeedForwardTorque", 1)


        self.init_pos           = 0.0
        self.init_vel           = 0.0
        self.current_pos        = 0.0


        motor_id    = 5
        self.motor_1 = TMotorManager_mit_can(motor_type='AK60-6', motor_ID=motor_id)
        self.motor_1.set_zero_position() # has a delay!
        time.sleep(1) # wait for the motor to zero (~1 second)
        self.motor_1.__enter__()
        time.sleep(1) # wait for the motor to zero (~1 second)
        self.motor_1.set_impedance_gains_real_unit_full_state_feedback(K=0.0, B=0.0)
        self.motor_1.update()
        self.init_pos = self.motor_1.get_output_angle_radians()

        self.muscle = MuscleModel(_a        = 5.0,
                                  _b        = 5.0,
                                  _beta     = 0.05,
                                  _init_pos = self.init_pos,
                                  number_motor = 1)


        ros_node_freq = 1000 #Hz Less is better for CAN bus [50, 100] Hz
        self.timer = self.create_timer((1/ros_node_freq), self.MainLoop)
        self.max_runtime = 15  # seconds


        self.LPF_vel = LPF(1.0)

        print("Starting position tracking demo. Press ctrl+C to quit.")
        self.start_time = self.get_clock().now().nanoseconds / 1e9  # Start time in seconds


    def MainLoop(self):
        current_time = self.get_clock().now().nanoseconds / 1e9  # Convert nanoseconds to seconds
        t = current_time - self.start_time  # Elapsed time since start

        # Sinusoidal position and velocity
        A_max = 0.75
        frequency = 3                                             # Frequency in Hz
        
        ramp_time = 2
        amplitude = A_max * (1 - np.exp(-t / ramp_time))                # Amplitude of position sine wave (radians)
        omega = 2 * np.pi * frequency                                   # Angular frequency
        
        pos = amplitude * np.sin(omega * t )  # Position (sine wave)
        # vel = amplitude * omega * np.cos(omega * t)  # Velocity (derivative of sine wave)

        # Online Adaptive Impedance Control
        self.vel_fb = self.LPF_vel.filter(self.motor_1.get_output_velocity_radians_per_second())
        self.muscle.calculate(pos_des = pos, 
                              pos_fb  = self.motor_1.get_output_angle_radians(),        # It has been minus init_pos in class
                              vel_fb  = self.vel_fb)
        


        # Set motion command
        # self.motor_1.set_output_angle_radians(pos = 0.0)
        self.motor_1.set_output_angle_radians(pos = pos)
        self.motor_1.set_output_velocity_radians_per_second(vel = self.muscle.vel_des)

        # Set impedance command
        self.motor_1.set_output_torque_newton_meters(torque = self.muscle.get_feedforward_force())
        self.motor_1.set_impedance_gains_real_unit_full_state_feedback(K = self.muscle.get_stiffness(), 
                                                                       B = self.muscle.get_damping())
        # self.motor_1.set_impedance_gains_real_unit_full_state_feedback(K = 1.0, B = 0.1)
        # self.motor_1.set_output_torque_newton_meters(torque = self.muscle.tau)

        # Send commnad to motor
        self.motor_1.update()

        # =========== ROS Publish =========== #
        self.msg_f32 = Float32MultiArray()
        # Joint angle
        self.current_pos = [np.float64(self.motor_1.get_output_angle_radians())]
        self.msg_f32.data = self.current_pos
        self.pub_q.publish(self.msg_f32)

        # Joint velocity
        self.msg_f32.data = [np.float64(self.vel_fb)]
        self.pub_q_dot.publish(self.msg_f32)

        # Joint angle command
        # self.msg_f32.data = [np.float64(self.muscle.x_r)]
        self.msg_f32.data = [np.float64(pos)]
        self.pub_qd.publish(self.msg_f32)

        # Joint velocity command
        self.msg_f32.data = [np.float64(self.muscle.vel_des)]
        self.pub_qd_dot.publish(self.msg_f32)
            
        # Joint angle error
        self.msg_f32.data = [np.float64(self.muscle.gen_track_error())]
        self.pub_qe.publish(self.msg_f32)

        # Joint torque
        self.msg_f32.data = [np.float64(self.motor_1.get_output_torque_newton_meters())]
        self.pub_tor.publish(self.msg_f32)

        self.msg_f32.data = [np.float64(self.muscle.K)]
        self.pub_k.publish(self.msg_f32)

        self.msg_f32.data = [np.float64(self.muscle.D)]
        self.pub_d.publish(self.msg_f32)

        self.msg_f32.data = [np.float64(self.muscle.F)]
        self.pub_f.publish(self.msg_f32)


        # Stop node after max_runtime
        if t >= self.max_runtime:
            print(f"Reached {self.max_runtime} seconds. Shutting down...")
            self.motor_1.power_off()
            rclpy.shutdown()
            self.destroy_node()
            return

def main(args=None):
    rclpy.init(args=args)
    node = MainNode()
    
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        print(" =============[Node was Terminated]============= ")
    finally:
        node.motor_1.power_off()
        rclpy.shutdown()
        node.destroy_node()

if __name__ == '__main__':
    main()