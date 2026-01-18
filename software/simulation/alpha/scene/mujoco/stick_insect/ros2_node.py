from rclpy.node import Node
from std_msgs.msg import Float32MultiArray

class StickInsectNode(Node):
    def __init__(self):
        super().__init__('stick_insect_controller')
        
        self.joint_cmd_sub  = self.create_subscription( Float32MultiArray, '/stick_insect/joint_angle_commands', self.joint_cmd_callback, 1)
        
        
        self.grf_pub                    = self.create_publisher( Float32MultiArray, '/stick_insect/foot_force_feedback', 1 )
        self.joint_angle_pub            = self.create_publisher( Float32MultiArray, '/stick_insect/joint_angle_fb', 1 )
        self.joint_velocity_pub         = self.create_publisher( Float32MultiArray, '/stick_insect/joint_velocity_fb', 1 )
        self.joint_stiffness_pub        = self.create_publisher( Float32MultiArray, '/stick_insect/joint_stiffness_fb', 1 )
        self.joint_damping_pub          = self.create_publisher( Float32MultiArray, '/stick_insect/joint_damping_fb', 1 )
        self.joint_torque_ff_pub        = self.create_publisher( Float32MultiArray, '/stick_insect/joint_torque_feedforward_fb', 1 )
        self.joint_torque_output_pub    = self.create_publisher( Float32MultiArray, '/stick_insect/joint_torque_output_fb', 1 )
        

        self.joint_cmd = {'TR': [0.0, 0.0, 0.0],
                          'CR': [0.0, 0.0, 0.0],
                          'FR': [0.0, 0.0, 0.0],
                          'TL': [0.0, 0.0, 0.0],
                          'CL': [0.0, 0.0, 0.0],
                          'FL': [0.0, 0.0, 0.0]}
        print("ROS 2 Node Started. Subscribed to /joint_commands")

    def joint_cmd_callback(self, msg):
        self.joint_cmd['TR'] = msg.data[0:3]
        self.joint_cmd['CR'] = msg.data[3:6]
        self.joint_cmd['FR'] = msg.data[6:9]
        self.joint_cmd['TL'] = msg.data[9:12]
        self.joint_cmd['CL'] = msg.data[12:15]
        self.joint_cmd['FL'] = msg.data[15:18]

    def publish_grf(self, forces):
            msg = Float32MultiArray()
            msg.data = [float(f) for f in forces]
            self.grf_pub.publish(msg)
    def publish_joint_angle(self, angles):
            msg = Float32MultiArray()
            msg.data = [float(a) for a in angles]
            self.joint_angle_pub.publish(msg)
    def publish_joint_velocity(self, velocities):
            msg = Float32MultiArray()
            msg.data = [float(v) for v in velocities]
            self.joint_velocity_pub.publish(msg)
    def publish_joint_stiffness(self, stiffnesses):
            msg = Float32MultiArray()
            msg.data = [float(s) for s in stiffnesses]
            self.joint_stiffness_pub.publish(msg)
    def publish_joint_damping(self, dampings):
            msg = Float32MultiArray()
            msg.data = [float(d) for d in dampings]
            self.joint_damping_pub.publish(msg)
    def publish_joint_torque_ff(self, torques):
            msg = Float32MultiArray()
            msg.data = [float(t) for t in torques]
            self.joint_torque_ff_pub.publish(msg)
    def publish_joint_torque_output(self, torques):
            msg = Float32MultiArray()
            msg.data = [float(t) for t in torques]
            self.joint_torque_output_pub.publish(msg)