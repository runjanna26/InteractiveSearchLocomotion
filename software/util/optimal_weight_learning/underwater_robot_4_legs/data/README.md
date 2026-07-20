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
    "data/pibb_logs/pibb_training_20260711_144156.json", 
    "data/pibb_logs/pibb_training_20260711_132535.json",           
    "data/pibb_logs/pibb_training_20260711_053503.json"  
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

# file: pibb_training_20260714_150136 (Rough Terrain)

RENDER = False
ROLLOUTS = 10  # Number of parallel rollouts per iteration
SIMULATION_STEPS =  2000  # E.g., 5 seconds at 0.005s timestep
ITERATIONS = 500
NOISE_VARIANCE_INIT = 0.0050
BASE_PARAM_INIT = 0.000
NUM_KERNELS = 20 
NUM_PARAMETERS = NUM_KERNELS * 8 # 20x8 = 160 parameters
CPG_PHI = 0.08

w1 = 5.0      # Distance weight
w2 = 2.0      # Instability weight
w4 = 5.0      # Y-Drift penalty weight

w_collision = 1000.0  # NEW: Collision penalty weight
w_ripple    = 0.0 
w_accel     = 0.0  

total_reward = (w1 * distance_walked) - (w2 * instability) - (w4 * drift_y) - (w_collision * avg_collision) - (w_ripple * avg_torque_ripple) - (w_accel * avg_joint_accel)

torque = np.clip(ctrl.get_torque(), -15.0, 15.0) 
self.data.ctrl[self.actuator_ids[name]] = torque

frictionloss="0.1" armature="0.05"  // Add this for all joint in robot.xml
contype="1" conaffinity="1"         // Add this for 2 last link of every leg

# file: pibb_training_20260715_154125

RENDER = False
ROLLOUTS = 12  # Number of parallel rollouts per iteration
SIMULATION_STEPS = 2000  # E.g., 5 seconds at 0.005s timestep
ITERATIONS = 500
NOISE_VARIANCE_INIT = 0.0010
BASE_PARAM_INIT = 0.000
NUM_KERNELS = 20 
NUM_PARAMETERS = NUM_KERNELS * 16 # 20x16 = 320 parameters for swimming
CPG_PHI = 0.08

w1 = 5.0      # Distance weight
w2 = 2.0      # Instability weight
w4 = 5.0      # Y-Drift penalty weight

w_collision = 1000.0  # NEW: Collision penalty weight
w_ripple    = 0.0 
w_accel     = 0.0  

total_reward = (w1 * distance_walked) - (w2 * instability) - (w4 * drift_y) - (w_collision * avg_collision) - (w_ripple * avg_torque_ripple) - (w_accel * avg_joint_accel)

torque = np.clip(ctrl.get_torque(), -15.0, 15.0) 
self.data.ctrl[self.actuator_ids[name]] = torque

frictionloss="0.1" armature="0.05"  // Add this for all joint in robot.xml
contype="1" conaffinity="1"         // Add this for 2 last link of every leg

## DRAG COEFFICIENTS
self.Cd_side = 6.0 # <--- INCREASE this to make it "thicker" 1.2 // Cylinder moving sideways (flat against flow)
self.Cd_end = 0.5  # <--- INCREASE this to make it "thicker" 0.8 // Cylinder moving end-on (like a needle)

self.linear_drag_coeff = 40.0 
self.angular_drag_coeff = 10.0 # Tune this if it spins too much or too little

target_density = 900 // Robot density to ensure robot floats (500 kg/m^3 = Light wood/plastic)

viscosity="2.0"

<inertial pos="0 -0.1 0.3" mass="5.0" diaginertia="0.01521 0.01521 9.453e-05" />
<geom type="capsule" size="0.1 0.2" pos="0 0 0" rgba="0 0 0 0" contype="0" conaffinity="0" group="3"/>
<geom quat="-0.707108 0.707105 0 2.44949e-07" type="mesh" mesh="underwater_insect_robot_base_coll_1" material="mat_legs" contype="1" conaffinity="1" group="1"/>

<!-- Leg 1 High Front leg: 0.140684 -0.026921 0.339098-->
<!-- Leg 1 Low Front leg: 0.140684 -0.026921 0.223809-->
<body name="FR_L1" pos="0.140684 -0.026921 0.339098" quat="0.707103 -0.00241547 -0.00241572 -0.707103">
<inertial pos="0.000123 0.068716 -0.038767" quat="0.499117 0.500952 -0.502851 0.497062" mass="0.431618" diaginertia="0.000458537 0.000422628 0.00025388" />
<joint name="FR_J1" pos="0 0 0" axis="0 0 1" damping="2.0" frictionloss="0.1" armature="0.05"/>
<geom type="capsule" size="0.02 0.05" pos="0 0 0" rgba="1 0 0 0.5" contype="0" conaffinity="0" group="3"/>


half_cycle = cpg_cycle_length // 2

for _ in range(half_cycle):
    # Manually step the FL and BR oscillators forward in time
    # before the MuJoCo simulation even begins.
    cpg_output['FL'] = cpg_modulated['FL'].modulate_cpg(CPG_PHI, 0.0, 1.0)
    cpg_output['BR'] = cpg_modulated['BR'].modulate_cpg(CPG_PHI, 0.0, 1.0)

# ===============================================================
# FULLY INDEPENDENT WEIGHT LOGIC (No Symmetry)
# ===============================================================
imitated_weights = {}

# We must loop through BOTH sides now, exactly matching the order
# that base_parameters was packed in the main loop!
joint_index = 0
for side in LEG_SIDE:       # ['R', 'L']
    for index in LEG_INDEX: # ['F', 'B']
        for joint in JOINT_NAMES: # [0, 1, 2, 3]
            
            start_idx = joint_index * NUM_KERNELS
            end_idx = start_idx + NUM_KERNELS
            
            # Extract the 20 weights for this specific independent joint
            extracted_weights = noisy_parameters[start_idx:end_idx]
            
            # Assign them directly to the unique dictionary key
            dict_key = f"{index}{side}{joint}"
            imitated_weights[dict_key] = extracted_weights
            
            joint_index += 1
# ===============================================================