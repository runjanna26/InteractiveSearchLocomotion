# PIBB find optimal pattern forr each  terrain

This is the first log that we it can learn:

# file: pibb_training_20260710_181135
configuration:
RENDER = False
ROLLOUTS = 10  # Number of parallel rollouts per iteration
SIMULATION_STEPS =  2500  # E.g., 5 seconds at 0.005s timestep
ITERATIONS = 350
NOISE_VARIANCE_INIT = 0.0010
BASE_PARAM_INIT = 0.000
NUM_KERNELS = 20 
NUM_PARAMETERS = NUM_KERNELS * 8 # 20x8 = 160 parameters
CPG_PHI = 0.03

w1 = 5.0      # Distance weight
w2 = 2.0      # Instability weight
w4 = 3.0      # Y-Drift penalty weight
w_ripple    = 0.0 
w_accel     = 0.0  
        
total_reward = (w1 * distance_walked) - (w2 * instability) - (w4 * drift_y) - (w_ripple * avg_torque_ripple) - (w_accel * avg_joint_accel)

torque = np.clip(ctrl.get_torque(), -10.0, 10.0) 
self.data.ctrl[self.actuator_ids[name]] = torque

# file: pibb_training_20260710_212046
configuration:
RENDER = False
ROLLOUTS = 10  # Number of parallel rollouts per iteration
SIMULATION_STEPS =  2500  # E.g., 5 seconds at 0.005s timestep
ITERATIONS = 500
NOISE_VARIANCE_INIT = 0.0010
BASE_PARAM_INIT = 0.000
NUM_KERNELS = 20 
NUM_PARAMETERS = NUM_KERNELS * 8 # 20x8 = 160 parameters
CPG_PHI = 0.03

w1 = 5.0      # Distance weight
w2 = 2.0      # Instability weight
w4 = 4.0      # Y-Drift penalty weight
w_ripple    = 0.0 
w_accel     = 0.0  
        
total_reward = (w1 * distance_walked) - (w2 * instability) - (w4 * drift_y) - (w_ripple * avg_torque_ripple) - (w_accel * avg_joint_accel)

torque = np.clip(ctrl.get_torque(), -10.0, 10.0) 
self.data.ctrl[self.actuator_ids[name]] = torque

# file: pibb_training_20260710_234228
configuration:
RENDER = False
ROLLOUTS = 10  # Number of parallel rollouts per iteration
SIMULATION_STEPS =  2000  # E.g., 5 seconds at 0.005s timestep
ITERATIONS = 500
NOISE_VARIANCE_INIT = 0.0010
BASE_PARAM_INIT = 0.000
NUM_KERNELS = 20 
NUM_PARAMETERS = NUM_KERNELS * 8 # 20x8 = 160 parameters
CPG_PHI = 0.05

w1 = 5.0      # Distance weight
w2 = 2.0      # Instability weight
w4 = 5.0      # Y-Drift penalty weight

w_ripple    = 0.0 
w_accel     = 0.0  

total_reward = (w1 * distance_walked) - (w2 * instability) - (w4 * drift_y) - (w_ripple * avg_torque_ripple) - (w_accel * avg_joint_accel)

torque = np.clip(ctrl.get_torque(), -15.0, 15.0) 
self.data.ctrl[self.actuator_ids[name]] = torque

# file: pibb_training_20260711_053503 (Best Walking!!)
configuration:

RENDER = False
ROLLOUTS = 10  # Number of parallel rollouts per iteration
SIMULATION_STEPS =  2000  # E.g., 5 seconds at 0.005s timestep
ITERATIONS = 500
NOISE_VARIANCE_INIT = 0.0010
BASE_PARAM_INIT = 0.000
NUM_KERNELS = 20 
NUM_PARAMETERS = NUM_KERNELS * 8 # 20x8 = 160 parameters
CPG_PHI = 0.05

w1 = 5.0      # Distance weight
w2 = 2.0      # Instability weight
w4 = 5.0      # Y-Drift penalty weight

w_collision = 2.0  # NEW: Collision penalty weight
w_ripple    = 0.0 
w_accel     = 0.0  

total_reward = (w1 * distance_walked) - (w2 * instability) - (w4 * drift_y) - (w_collision * avg_collision) - (w_ripple * avg_torque_ripple) - (w_accel * avg_joint_accel)
    

for i in range(self.data.ncon):
    contact = self.data.contact[i]
    
    # Get the ID of the main body that each colliding geometry belongs to
    body1_id = self.model.geom_bodyid[contact.geom1]
    body2_id = self.model.geom_bodyid[contact.geom2]
    
    # Convert those IDs into string names (e.g., 'FR_calf', 'FR_thigh')
    body1_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body1_id)
    body2_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body2_id)
    
    # Safely check if they have names
    if body1_name and body2_name:
        # List your leg prefixes
        for prefix in ['FR_', 'FL_', 'BR_', 'BL_']:
            # If BOTH colliding bodies start with the exact same prefix...
            if body1_name.startswith(prefix) and body2_name.startswith(prefix):
                # ...it is an intra-leg collision! 
                # Add a flat penalty for this frame.
                step_collision += 1.0
                break # Stop checking prefixes

torque = np.clip(ctrl.get_torque(), -12.0, 12.0) 
self.data.ctrl[self.actuator_ids[name]] = torque

frictionloss="0.1" armature="0.05"  // Add this for all joint in robot.xml
contype="1" conaffinity="1"         // Add this for 2 last link of every leg

