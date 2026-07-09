import json
import time
import numpy as np
from script import StickInsectEnv
from cpg_rbf.cpg_so2 import CPG_SO2, CPG_LOCO
from cpg_rbf.rbf import RBF

'''
How to use it to visualize learning:
If you want to see exactly how your robot figured out its gait, you can run this script multiple times and just change the TARGET_ITERATION variable:

Set TARGET_ITERATION = 5 to watch the robot likely flail and fall over.

Set TARGET_ITERATION = 150 to watch it figure out how to stand up and crawl clumsily.

Set TARGET_ITERATION = 349 to watch the highly optimized, smooth walking pattern.
'''

# ======================================================
# CONFIGURATION
# ======================================================
LOG_FILE = "data/pibb_logs/pibb_training_20260709_145905.json" # <-- Paste your actual log filename here
TARGET_ITERATION = 30    # Set to -1 for the last iteration, or a specific number (e.g., 150)
SIMULATION_STEPS = 5000

LEG_SIDE    = ['R', 'L']
LEG_INDEX   = ["F", "B"]
JOINT_NAMES = [0, 1, 2, 3]
NUM_KERNELS = 20 
STANDING_POSE = {
    'FR': [ np.pi/6, 0.0,  np.pi/9, -2*np.pi/4],
    'BR': [-np.pi/6, 0.0,  np.pi/3, -2*np.pi/3],
    'FL': [-np.pi/6, 0.0,  np.pi/9, -2*np.pi/4],
    'BL': [ np.pi/6, 0.0,  np.pi/3, -2*np.pi/3]
}
if __name__ == "__main__":
    print(f"Loading log file: {LOG_FILE}")
    
    # 1. Load the history array from the log file
    with open(LOG_FILE, "r") as f:
        log_history = json.load(f)

    # 2. Extract the specific iteration you want to replay
    if TARGET_ITERATION == -1:
        # Grab the very last entry in the list
        selected_data = log_history[-1]
    else:
        # Search the log for the exact iteration number you requested
        selected_data = next((item for item in log_history if item["iteration"] == TARGET_ITERATION), None)
        if selected_data is None:
            print(f"Error: Iteration {TARGET_ITERATION} not found in the log.")
            exit()

    trained_weights = selected_data["best_parameters"]
    fitness_score = selected_data["metrics"]["max_fitness"]
    
    print(f"Replaying Iteration {selected_data['iteration']} | Max Fitness: {fitness_score:.4f}")

    # 3. Slice the weights for your joints (Exact same as before)
    # imitated_weights = {}
    # joint_keys = [(side, index, joint) for side in LEG_SIDE for index in LEG_INDEX for joint in JOINT_NAMES]
    
    # for i, (side, index, joint) in enumerate(joint_keys):
    #     start_idx = i * NUM_KERNELS
    #     end_idx = start_idx + NUM_KERNELS
    #     dict_key = f'{index}{side}{joint}'
    #     imitated_weights[dict_key] = trained_weights[start_idx:end_idx]

    # Assuming 'best_parameters' is the variable holding the 160 weights loaded from your JSON
    imitated_weights = {}
    joint_index = 0
    
    for index in LEG_INDEX: 
        for joint in JOINT_NAMES:
            start_idx = joint_index * NUM_KERNELS
            end_idx = start_idx + NUM_KERNELS
            
            # Extract the 20 weights for this specific joint
            extracted_weights = trained_weights[start_idx:end_idx]
            
            # MIRROR them exactly to both the RIGHT and LEFT sides!
            imitated_weights[f"{index}R{joint}"] = extracted_weights
            imitated_weights[f"{index}L{joint}"] = extracted_weights 
            
            joint_index += 1

    # 3. Initialize Neural Networks
    cpg = CPG_SO2()
    rbf = RBF(nc=NUM_KERNELS)

    cpg_one_cycle = cpg.generate_cpg_one_cycle(0.05)
    rbf.construct_kernels_with_cpg_one_cycle(
        cpg_one_cycle['out0_cpg_one_cycle'][:], 
        cpg_one_cycle['out1_cpg_one_cycle'][:], 
        len(cpg_one_cycle['out0_cpg_one_cycle'])
    )

    cpg_modulated = {}
    cpg_output = {}
    cpg_mod_cmd = {}

    for side in LEG_SIDE:
        for index in LEG_INDEX:
            cpg_modulated[f'{index}{side}'] = CPG_LOCO()
            cpg_output[f'{index}{side}']    = cpg_modulated[f'{index}{side}'].modulate_cpg(0.05, 0.0, 1.0)
            cpg_mod_cmd[f'{index}{side}']   = {'phi': 0.05, 'pause_input': 0.0, 'rewind_input': 1.0}

    # 4. Initialize Environment WITH Rendering (and ROS if you want to record bags)
    env = StickInsectEnv(enable_ros=True, render=True) 
    env.reset()
    
    print("Starting replay...")
    
    # 5. Run the Simulation Loop
    for step in range(SIMULATION_STEPS):
        # Update CPGs
        for side in LEG_SIDE:
            for index in LEG_INDEX:
                cpg_output[f'{index}{side}'] = cpg_modulated[f'{index}{side}'].modulate_cpg(
                    cpg_mod_cmd[f'{index}{side}']['phi'], 
                    cpg_mod_cmd[f'{index}{side}']['pause_input'], 
                    cpg_mod_cmd[f'{index}{side}']['rewind_input']
                )
        
        # Create a FLAT dictionary for the environment targets (e.g. {'FR_J1': 0.1, 'FR_J2': 0.5})
        env_targets = {}
        
        # Update RBFs
        for side in LEG_SIDE:
            for index in LEG_INDEX:
                for joint in JOINT_NAMES:  # joint is 0, 1, 2, or 3
                    
                    # 1. Calculate the rhythmic target angle from the neural network
                    network_output = rbf.regenerate_target_traj(
                        cpg_output[f'{index}{side}']['cpg_output_0'], 
                        cpg_output[f'{index}{side}']['cpg_output_1'],
                        imitated_weights[f'{index}{side}{joint}']
                    )
                    
                    # 2. Add it to your standing pose
                    # f'{index}{side}' gives 'FR' and 'joint' acts as the list index [0]
                    baseline_angle = STANDING_POSE[f'{index}{side}'][joint]
                    target_angle = network_output + baseline_angle
                    
                    # 3. Store it using the exact string format your MuJoCo XML actuators use
                    actuator_name = f"{index}{side}_J{joint+1}"  # +1 because your actuators are likely 1-indexed
                    env_targets[actuator_name] = target_angle
            
        # Pass the flat dictionary to the environment
        env.step(env_targets)
        
        # Optional: Add a tiny sleep to slow down the viewer to real-time 
        # if your computer calculates physics much faster than real-time
        # time.sleep(0.005) 

    print("Replay finished.")