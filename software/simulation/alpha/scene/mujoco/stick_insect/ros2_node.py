from rclpy.node import Node
from std_msgs.msg import Float32MultiArray

class StickInsectNode(Node):
    def __init__(self):
        super().__init__('stick_insect_controller')
        
        # Subscribe to standard JointState message
        # Topic: /joint_commands
        self.subscription = self.create_subscription( Float32MultiArray, '/stick_insect/joint_angle_commands', self.JointAnglesCmd, 1)
        self.grf_publisher = self.create_publisher( Float32MultiArray, '/stick_insect/foot_force_feedback', 1 )
        

        self.joint_cmd = {'TR': [0.0, 0.0, 0.0],
                          'CR': [0.0, 0.0, 0.0],
                          'FR': [0.0, 0.0, 0.0],
                          'TL': [0.0, 0.0, 0.0],
                          'CL': [0.0, 0.0, 0.0],
                          'FL': [0.0, 0.0, 0.0]}
        print("ROS 2 Node Started. Subscribed to /joint_commands")

    def JointAnglesCmd(self, msg):
        self.joint_cmd['TR'] = msg.data[0:3]
        self.joint_cmd['CR'] = msg.data[3:6]
        self.joint_cmd['FR'] = msg.data[6:9]
        self.joint_cmd['TL'] = msg.data[9:12]
        self.joint_cmd['CL'] = msg.data[12:15]
        self.joint_cmd['FL'] = msg.data[15:18]

    def publish_grf(self, forces):
            msg = Float32MultiArray()
            msg.data = [float(f) for f in forces]
            self.grf_publisher.publish(msg)