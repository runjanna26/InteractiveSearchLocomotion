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
LOG_FILE = "data/pibb_logs/pibb_training_20260723_050740.json" # <-- Paste your actual log filename here
TARGET_ITERATION = -1  # Set to -1 for the last iteration, or a specific number (e.g., 150)
SIMULATION_STEPS = 100000000
CPG_PHI = 0.05

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

    
    # ===============================================================
    # RESIDUAL SYMMETRY LOGIC (Reconstructs Left offsets from Prior)
    # ===============================================================
    print("Loading Prior Knowledge to reconstruct Left offsets...")
    prior_knowledge = np.load('imitated_diving_beetle_swim_forward_weights_20_kernels.npz')

    imitated_weights = {}
    
    joint_index = 0
    for index in LEG_INDEX: 
        for joint in JOINT_NAMES:
            start_idx = joint_index * NUM_KERNELS
            end_idx = start_idx + NUM_KERNELS
            
            # 1. Get the Original Priors first so we can use them for the freeze
            right_key = f"{index}R{joint}"
            left_key = f"{index}L{joint}"
            
            right_prior = np.array(prior_knowledge[right_key])
            left_prior = np.array(prior_knowledge[left_key])
            
            # ==========================================
            # 🚨 THE FIX: FREEZE JOINT 1 
            # ==========================================
            if joint == 1:
                # FREEZE: Ignore the JSON log. Force it to be the original prior.
                learned_weights = np.copy(right_prior)
            else:
                # LEARN: Get the optimized weights from the JSON log
                learned_weights = np.array(trained_weights[start_idx:end_idx])
            
            # Calculate the exact difference (Left - Right)
            offset_weights = left_prior - right_prior
            
            # 3. Assign the pure weights to the RIGHT side
            imitated_weights[right_key] = learned_weights
            
            # 4. Assign the Weights + Original Offset to the LEFT side
            imitated_weights[left_key] = learned_weights + offset_weights
            
            joint_index += 1
    # ===============================================================

    # # ===============================================================
    # # WEIGHT SYMMETRY LOGIC
    # # ===============================================================
    # imitated_weights = {}
    
    # # 1. We only loop through the Front ('F') and Back ('B') indices
    # joint_index = 0
    # for index in LEG_INDEX: 
    #     for joint in JOINT_NAMES:
    #         start_idx = joint_index * NUM_KERNELS
    #         end_idx = start_idx + NUM_KERNELS
            
    #         # Extract the 20 weights for this specific joint
    #         extracted_weights = trained_weights[start_idx:end_idx]
            
    #         # 2. Assign these weights to the RIGHT side
    #         right_key = f"{index}R{joint}"
    #         imitated_weights[right_key] = extracted_weights
            
    #         # 3. MIRROR them exactly to the LEFT side!
    #         left_key = f"{index}L{joint}"
    #         imitated_weights[left_key] = extracted_weights
            
    #         joint_index += 1
    # ===============================================================

    # # ===============================================================
    # # FULLY INDEPENDENT WEIGHT LOGIC (No Symmetry)
    # # ===============================================================
    # imitated_weights = {}
    
    # # We must loop through BOTH sides now, exactly matching the order
    # # that base_parameters was packed in the main loop!
    # joint_index = 0
    # for side in LEG_SIDE:       # ['R', 'L']
    #     for index in LEG_INDEX: # ['F', 'B']
    #         for joint in JOINT_NAMES: # [0, 1, 2, 3]
                
    #             start_idx = joint_index * NUM_KERNELS
    #             end_idx = start_idx + NUM_KERNELS
                
    #             # Extract the 20 weights for this specific independent joint
    #             extracted_weights = trained_weights[start_idx:end_idx]
                
    #             # Assign them directly to the unique dictionary key
    #             dict_key = f"{index}{side}{joint}"
    #             imitated_weights[dict_key] = extracted_weights
                
    #             joint_index += 1
    # # ===============================================================

    # 3. Initialize Neural Networks
    cpg = CPG_SO2()
    rbf = RBF(nc=NUM_KERNELS)

    cpg_one_cycle = cpg.generate_cpg_one_cycle(CPG_PHI)
    rbf.construct_kernels_with_cpg_one_cycle(
        cpg_one_cycle['out0_cpg_one_cycle'][:], 
        cpg_one_cycle['out1_cpg_one_cycle'][:], 
        len(cpg_one_cycle['out0_cpg_one_cycle'])
    )

    cpg_cycle_length = len(cpg_one_cycle['out0_cpg_one_cycle'][:])

    cpg_modulated = {}
    cpg_output = {}
    cpg_mod_cmd = {}

    for side in LEG_SIDE:
        for index in LEG_INDEX:
            cpg_modulated[f'{index}{side}'] = CPG_LOCO()
            cpg_output[f'{index}{side}']    = cpg_modulated[f'{index}{side}'].modulate_cpg(CPG_PHI, 0.0, 1.0)
            cpg_mod_cmd[f'{index}{side}']   = {'phi': CPG_PHI, 'pause_input': 0.0, 'rewind_input': 1.0}


    half_cycle = cpg_cycle_length // 2
    
    for _ in range(half_cycle):
        # Manually step the FL and BR oscillators forward in time
        # before the MuJoCo simulation even begins.
        cpg_output['FL'] = cpg_modulated['FL'].modulate_cpg(CPG_PHI, 0.0, 1.0)
        cpg_output['BR'] = cpg_modulated['BR'].modulate_cpg(CPG_PHI, 0.0, 1.0)

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
                    
                    # ==========================================
                    # NEW: MECHANICAL INVERSION FOR LEFT HIP
                    # ==========================================
                    # Because the Left hip (J0) axis is mirrored in the XML, 
                    # we must invert the network output to make it swing the 
                    # same physical direction as the Right hip.
                    if side == 'L' and joint == 0:
                        network_output = -network_output 
                    
                    # 2. Add it to your standing pose
                    baseline_angle = STANDING_POSE[f'{index}{side}'][joint]
                    target_angle =  network_output
                    
                    # 3. Store it using the exact string format your MuJoCo XML actuators use
                    actuator_name = f"{index}{side}_J{joint+1}"  
                    env_targets[actuator_name] = target_angle
            
        # Pass the flat dictionary to the environment
        env.step(env_targets)
        
        # Optional: Add a tiny sleep to slow down the viewer to real-time 
        # if your computer calculates physics much faster than real-time
        # time.sleep(0.005) 

    print("Replay finished.")