import json
import time
import numpy as np
from script import StickInsectEnv
from cpg_rbf.cpg_so2 import CPG_SO2, CPG_LOCO
from cpg_rbf.rbf import RBF

'''
How to use it to visualize learning:
Set TARGET_ITERATION = 5 to watch the robot flail.
Set TARGET_ITERATION = 150 to watch it figure out how to crawl.
Set TARGET_ITERATION = -1 to watch the final optimized pattern.
'''

# ======================================================
# CONFIGURATION
# ======================================================
LOG_FILE = "data/pibb_logs/pibb_training_20260802_183609.json" # <-- Paste your actual log filename here
TARGET_ITERATION = -1
SIMULATION_STEPS = 100000
CPG_PHI = 0.05

LEG_SIDE    = ['R', 'L']
LEG_INDEX   = ["F", "B"]
JOINT_NAMES = [0, 1, 2]
NUM_KERNELS = 20 

STANDING_POSE = {
    'FR': [np.deg2rad( 0),  np.deg2rad(-55), np.deg2rad(100)],
    'BR': [np.deg2rad( 0),  np.deg2rad(-55), np.deg2rad(100)],
    'FL': [np.deg2rad( 0),  np.deg2rad(-55), np.deg2rad(100)],
    'BL': [np.deg2rad( 0),  np.deg2rad(-55), np.deg2rad(100)]
}

if __name__ == "__main__":
    print(f"Loading log file: {LOG_FILE}")
    
    # 1. Load the history array from the log file
    with open(LOG_FILE, "r") as f:
        log_history = json.load(f)

    # 2. Extract the specific iteration you want to replay
    if TARGET_ITERATION == -1:
        selected_data = log_history[-1]
    else:
        selected_data = next((item for item in log_history if item["iteration"] == TARGET_ITERATION), None)
        if selected_data is None:
            print(f"Error: Iteration {TARGET_ITERATION} not found in the log.")
            exit()

    trained_weights = selected_data["best_parameters"]
    fitness_score = selected_data["metrics"]["max_fitness"]
    
    print(f"Replay Iteration: {selected_data['iteration']} | Max Fitness: {fitness_score:.4f}")

    # ===============================================================
    # UNIFIED LEG WEIGHT LOGIC (All 4 legs share the same weights)
    # ===============================================================
    imitated_weights = {}
    
    # We only extract 3 joints (0, 1, 2) from the noisy_parameters array
    for joint in JOINT_NAMES: 
        start_idx = joint * NUM_KERNELS
        end_idx = start_idx + NUM_KERNELS
        
        if joint == 0:
            # 🚨 FREEZE: Ignore PIBB's exploration noise.
            extracted_weights = np.zeros(NUM_KERNELS)
        else:
            # LEARN: Use PIBB's noisy parameters for J1 and J2
            extracted_weights = trained_weights[start_idx:end_idx]
        
        # Now, broadcast this EXACT same set of weights to all four legs!
        for side in LEG_SIDE:       # ['R', 'L']
            for index in LEG_INDEX: # ['F', 'B']
                dict_key = f"{index}{side}{joint}"
                imitated_weights[dict_key] = extracted_weights
    # ===============================================================

    # # ===============================================================
    # # FULLY INDEPENDENT WEIGHT LOGIC (Scratch Learning - Joint 0 Frozen)
    # # ===============================================================
    # imitated_weights = {}
    
    # # Slice the flat 240-length array into 12 distinct joints
    # joint_index = 0
    # for side in LEG_SIDE:       # ['R', 'L']
    #     for index in LEG_INDEX: # ['F', 'B']
    #         for joint in JOINT_NAMES: # [0, 1, 2]
                
    #             start_idx = joint_index * NUM_KERNELS
    #             end_idx = start_idx + NUM_KERNELS
                
    #             if joint == 0:
    #                 # 🚨 FREEZE: Ignore the trained weights. 
    #                 # Force to zero so it just holds the standing pose.
    #                 extracted_weights = np.zeros(NUM_KERNELS)
    #             else:
    #                 # REPLAY: Use the optimized weights for J1 and J2
    #                 extracted_weights = trained_weights[start_idx:end_idx]
                
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

    # ===============================================================
    # CPG PHASE SHIFT LOGIC (THE TROT GAIT)
    # ===============================================================
    # Note: Make sure this matches your main_learning.py EXACTLY. 
    # If it was active in training, it must be active here!
    half_cycle = cpg_cycle_length // 2
    for _ in range(half_cycle):
        cpg_output['FL'] = cpg_modulated['FL'].modulate_cpg(CPG_PHI, 0.0, 1.0)
        cpg_output['BR'] = cpg_modulated['BR'].modulate_cpg(CPG_PHI, 0.0, 1.0)
    # ===============================================================

    # 4. Initialize Environment
    env = StickInsectEnv(enable_ros=True, render=True) 
    env.reset()
    
    print("Starting replay...")
    
    # 5. Run the Simulation Loop
    for step in range(SIMULATION_STEPS):
        step_start = time.perf_counter() 

        for side in LEG_SIDE:
            for index in LEG_INDEX:
                cpg_output[f'{index}{side}'] = cpg_modulated[f'{index}{side}'].modulate_cpg(
                    cpg_mod_cmd[f'{index}{side}']['phi'], 
                    cpg_mod_cmd[f'{index}{side}']['pause_input'], 
                    cpg_mod_cmd[f'{index}{side}']['rewind_input']
                )
        
        env_targets = {}
        
        for side in LEG_SIDE:
            for index in LEG_INDEX:
                for joint in JOINT_NAMES: 
                    
                    network_output = rbf.regenerate_target_traj(
                        cpg_output[f'{index}{side}']['cpg_output_0'], 
                        cpg_output[f'{index}{side}']['cpg_output_1'],
                        imitated_weights[f'{index}{side}{joint}']
                    )
                    
                    baseline_angle = STANDING_POSE[f'{index}{side}'][joint]
                    target_angle = baseline_angle + network_output
                    
                    actuator_name = f"{index}{side}_J{joint+1}"  
                    env_targets[actuator_name] = target_angle
            
        env.step(env_targets)
        
        target_duration = env.model.opt.timestep
        while (time.perf_counter() - step_start) < target_duration:
            pass # Busy-wait ensures perfect timing

    print("Replay finished.")