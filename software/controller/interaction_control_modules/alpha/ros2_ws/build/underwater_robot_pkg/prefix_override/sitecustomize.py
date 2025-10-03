import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/runj/BRAIN/InteractiveSearchLocomotion/software/controller/interaction_control_modules/alpha/ros2_ws/install/underwater_robot_pkg'
