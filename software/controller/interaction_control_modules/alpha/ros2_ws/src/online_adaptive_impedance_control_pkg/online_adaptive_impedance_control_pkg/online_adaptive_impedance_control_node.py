#!/usr/bin/env python3
# import can
import time
import numpy as np
import rclpy
from rclpy.qos import ReliabilityPolicy, QoSProfile
from rclpy.node import Node
from rclpy.executors import ExternalShutdownException
from std_msgs.msg import Float64MultiArray, Float32MultiArray, UInt16MultiArray, Int16MultiArray, UInt8
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Joy,Imu

from tutorial_interfaces.srv import GetInitialPosition    


from mit_can import TMotorManager_mit_can


JointAngleCmd = []
axes_js = (0,0,0,0,0,0,0,0)
btn_js = (0,0,0,0,0,0,0,0)
class MinimalPublisher(Node):
    def __init__(self):
        super().__init__('robot_node')
        # self.pub_q        = self.create_publisher (Float32MultiArray, "/HERO/LegModule/JointAngles", 1)
        # self.pub_q_dot    = self.create_publisher (Float32MultiArray, "/HERO/LegModule/JointVelocities", 1)
        # self.pub_q_ddot   = self.create_publisher (Float32MultiArray, "/HERO/LegModule/JointAccelerations", 1)
        # self.pub_tor      = self.create_publisher (Float32MultiArray, "/HERO/LegModule/JointTorques", 1)
        # self.pub_k        = self.create_publisher (Float32MultiArray, "/HERO/LegModule/JointImpedances/Stiffness", 1)
        # self.pub_d        = self.create_publisher (Float32MultiArray, "/HERO/LegModule/JointImpedances/Damping", 1)
        # self.pub_tff      = self.create_publisher (Float32MultiArray, "/HERO/LegModule/JointImpedances/TorqueFeedForward", 1)

        # # Debugging Topics
        # self.pub_q_err    = self.create_publisher (Float32MultiArray, "/HERO/LegModule/JointAnglesError", 1)
        # self.pub_beta     = self.create_publisher (Float32MultiArray, "/HERO/LegModule/JointImpedances/beta", 1)

        # self.sub_qd_cmd     = self.create_subscription (Float32MultiArray, "/HERO/LegModule/JointCommands", self.JointCmdCallback, 1)
        # self.sub_jc         = self.create_subscription (Float32MultiArray, "/HERO/LegModule/JointCompliance", self.JointComplianceCallback, 1)
        # self.sub_qdd_cmd    = self.create_subscription (Float32MultiArray, "/HERO/LegModule/JointVelocityCommands", self.JointVelocityCmdCallback, 1)

        self.motor              = []
        self.muscle             = []
        self.init_pos           = [0.0, 0.0, 0.0, 0.0]
        self.current_pos        = [0, 0, 0, 0]
        
        self.JointAngleCmd      = [0, 0, 0, 0]
        self.JointVelocityCmd   = [0, 0, 0, 0]
        self.JointCompliance    = [0, 0, 0, 0]

        for i in range(4):
            self.motor.append(TMotorManager_mit_can(motor_type='AK60-6', motor_ID=i+1))
            self.motor[i].set_zero_position()
            time.sleep(3)
            self.motor[i].set_impedance_gains_real_unit_full_state_feedback(K=0.0, B=0.0)
            self.motor[i].__enter__()
            # self.motor[i].update()
            self.init_pos[i] = self.motor[i].get_output_angle_radians()
            # self.muscle.append(MuscleModel(_init_pos=self.init_pos[i]))
        print('initial position', [self.muscle[0].pos_init, self.muscle[1].pos_init, self.muscle[2].pos_init, self.muscle[3].pos_init])
        time.sleep(1.5)
        self.current_pos = self.init_pos

        self.now_time = time.time()
        self.prev_time = time.time()

        ros_node_freq = 100 #Hz Less is better for CAN bus [50, 100] Hz
        self.timer = self.create_timer((1/ros_node_freq), self.MainLoop)
        print("Press ctrl+C to quit.")

    # Get initial joint position service [update the current position] 
    def get_initial_position(self, request, response):
        response.init_positions = self.current_pos
        return response


    # Main Loop
    def MainLoop(self):

        #=====================[Test motor]=====================
        # [x] test muscle model with safety
        # [x] fast dynamic torque to evaluate stability
        # Optimal Compliance
        # self.muscle[0].a = 0.2
        # self.muscle[0].b = 10.0
        # Very Compliance
        # self.muscle[0].a = 35.0
        # self.muscle[0].b = 1.0

        # tau = self.muscle_1.gen_torque(0.0, self.motor[0].get_output_angle_radians()-self.init_pos[0], self.motor[0].get_output_velocity_radians_per_second())
        # self.motor[0].set_output_torque_newton_meters(tau)
        # self.motor[0].update()


        # [x] test friction compensate (work well)
        # tau = self.muscle[0].gen_torque_friction_compensate(self.motor[0].get_output_velocity_radians_per_second())
        # self.motor[0].set_output_torque_newton_meters(tau)
        # self.motor[0].update()

        # [x] test backdrive torque 0.5 Nm
        # self.motor[0].set_output_torque_newton_meters(0.5)
        # self.motor[0].update()

        # [x] Test Safe Torque Off (STO) (if it exceeding the MIT_params it will be power_off motors)
        # if self.muscle[0].get_error_status():
        #     self.motor[0].power_off()

        # [x] Adjust MIT_params max min of vel,tor,pos (cannot used because they use MIT_params to calculate command)
        
        #============================================================
        # Calculate muscle-like impedance/force of joint --> pd controller for now
        # self.muscle[0].cal_pd(pos_des = self.JointAngleCmd[0], 
        #                           pos_fb = self.motor[0].get_output_angle_radians(),        # It has been minus init_pos in class
        #                           vel_des = self.JointVelocityCmd[0], 
        #                           vel_fb = self.motor[0].get_output_velocity_radians_per_second(),
        #                           compliance_cmd = self.JointCompliance[0])
        
        # self.muscle[1].cal_pd(pos_des = self.JointAngleCmd[1], 
        #                           pos_fb = self.motor[1].get_output_angle_radians(),        # It has been minus init_pos in class
        #                           vel_des = self.JointVelocityCmd[1], 
        #                           vel_fb = self.motor[1].get_output_velocity_radians_per_second(),
        #                           compliance_cmd = self.JointCompliance[1])
    
        # self.muscle[2].cal_pd(pos_des = self.JointAngleCmd[2], 
        #                           pos_fb = self.motor[2].get_output_angle_radians(),        # It has been minus init_pos in class
        #                           vel_des = self.JointVelocityCmd[2], 
        #                           vel_fb = self.motor[2].get_output_velocity_radians_per_second(),
        #                           compliance_cmd = self.JointCompliance[2])
        
        # self.muscle[3].cal_pd(pos_des = self.JointAngleCmd[3], 
        #                           pos_fb = self.motor[3].get_output_angle_radians(),        # It has been minus init_pos in class
        #                           vel_des = self.JointVelocityCmd[3], 
        #                           vel_fb = self.motor[3].get_output_velocity_radians_per_second(),
        #                           compliance_cmd = self.JointCompliance[3])

        # # Send the packages to can bus
        # self.motor[0].set_impedance_gains_real_unit_full_state_feedback(K = self.muscle[0].get_k(), B = self.muscle[0].get_d())
        # self.motor[1].set_impedance_gains_real_unit_full_state_feedback(K = self.muscle[1].get_k(), B = self.muscle[1].get_d())
        # self.motor[2].set_impedance_gains_real_unit_full_state_feedback(K = self.muscle[2].get_k(), B = self.muscle[2].get_d())
        # self.motor[3].set_impedance_gains_real_unit_full_state_feedback(K = self.muscle[3].get_k(), B = self.muscle[3].get_d())

        # self.motor[0].set_output_angle_radians(pos = self.JointAngleCmd[0])
        # self.motor[1].set_output_angle_radians(pos = self.JointAngleCmd[1])
        # self.motor[2].set_output_angle_radians(pos = self.JointAngleCmd[2])
        # self.motor[3].set_output_angle_radians(pos = self.JointAngleCmd[3])

        # self.motor[0].set_output_velocity_radians_per_second(vel = self.JointVelocityCmd[0])
        # self.motor[1].set_output_velocity_radians_per_second(vel = self.JointVelocityCmd[1])
        # self.motor[2].set_output_velocity_radians_per_second(vel = self.JointVelocityCmd[2])
        # self.motor[3].set_output_velocity_radians_per_second(vel = self.JointVelocityCmd[3])

        # self.motor[0].set_output_torque_newton_meters(torque = 0.0)
        # self.motor[1].set_output_torque_newton_meters(torque = 0.0)
        # self.motor[2].set_output_torque_newton_meters(torque = 0.0)
        # self.motor[3].set_output_torque_newton_meters(torque = 0.0)

        # # Update command and read sensory feedback
        # self.motor[0].update()
        # self.motor[1].update()
        # self.motor[2].update()
        # self.motor[3].update()

        # Print
        # print(self.motor[0])
        # print(self.motor[1])
        # print(self.motor[2])
        # print(self.motor[3])
        #============================================================

        # Safety Stop: Detect Acceleration
        # if (np.abs(self.motor[0].get_output_acceleration_radians_per_second_squared()) > 500.0) or (np.abs(self.motor[1].get_output_acceleration_radians_per_second_squared()) > 500.0) or (np.abs(self.motor[2].get_output_acceleration_radians_per_second_squared()) > 500.0) or (np.abs(self.motor[3].get_output_acceleration_radians_per_second_squared()) > 500.0):
        #     self.motor[0].power_off()
        #     self.motor[1].power_off()
        #     self.motor[2].power_off()
        #     self.motor[3].power_off()
        
        # # =========== ROS publish ===========
        # self.msg_float32 = Float32MultiArray()
        
        # # print()
        # # JointAngles
        # self.current_pos = [np.float(m.get_output_angle_radians() - self.init_pos[i]) for i, m in enumerate(self.motor)]
        # self.msg_float32.data = self.current_pos
        # self.pub_q.publish(self.msg_float32)
        # ## JointErrors
        # self.msg_float32.data = [np.float(m.cal_pos_error()) for m in self.muscle]
        # self.pub_q_err.publish(self.msg_float32)
        # ## JointVelocities
        # self.msg_float32.data = [np.float(m.get_output_velocity_radians_per_second()) for m in self.motor]
        # self.pub_q_dot.publish(self.msg_float32)
        # ## JointAccelerations
        # # self.msg_float32.data = [np.float(m.get_output_acceleration_radians_per_second_squared()) for m in self.motor]
        # # self.pub_q_ddot.publish(self.msg_float32)
        # ## JointTorques
        # self.msg_float32.data = [np.float(m.get_output_torque_newton_meters()) for m in self.motor]
        # self.pub_tor.publish(self.msg_float32)
        # ## Stiffness
        # self.msg_float32.data = [np.float(m.get_k()) for m in self.muscle]
        # self.pub_k.publish(self.msg_float32)
        # ## Damping
        # self.msg_float32.data = [np.float(m.get_d()) for m in self.muscle]
        # self.pub_d.publish(self.msg_float32)
        # ## Torque Feed Forward
        # self.msg_float32.data = [np.float(m.get_tff()) for m in self.muscle]
        # self.pub_tff.publish(self.msg_float32)
        # ## beta 
        # self.msg_float32.data = [np.float(m.get_beta()) for m in self.muscle]
        # self.pub_beta.publish(self.msg_float32)
        
        # print(time.time() - self.prev_time)
        self.prev_time =  self.now_time
        


# def printSensor(motor_obj):
#     print('=====[', motor_obj.device_info_string(), ']=====')
#     print('Position: %.3f (rad)'% motor_obj.get_output_angle_radians())
#     print('Velocity: %.3f (rad/s)'% motor_obj.get_output_velocity_radians_per_second())
#     print('Acceleration: %.3f (rad/s^2)'% motor_obj.get_output_acceleration_radians_per_second_squared())
#     print('Torque: %.3f (Nm)'% motor_obj.get_output_torque_newton_meters())
#     print('Temperature: %.3f (C)'% motor_obj.get_temperature_celsius())


def main(args=None):
    rclpy.init(args=args)
    minimal_publisher = MinimalPublisher()
    
    try:
        rclpy.spin(minimal_publisher)
    except (KeyboardInterrupt, ExternalShutdownException):
        print("=============Robot [Terminated]=============")
    finally:
        minimal_publisher.motor[0].power_off()
        minimal_publisher.motor[1].power_off()
        minimal_publisher.motor[2].power_off()
        minimal_publisher.motor[3].power_off()
        rclpy.shutdown()
        minimal_publisher.destroy_node()

if __name__ == '__main__':
    main()