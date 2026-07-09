import json
import time
import numpy as np
from script import StickInsectEnv
from cpg_rbf.cpg_so2 import CPG_SO2, CPG_LOCO
from cpg_rbf.rbf import RBF

'''
How to use this:
Ensure your training script has saved at least one checkpoint into data/checkpoints/.

Run this script normally: python3 replay.py

The MuJoCo viewer will pop up, apply your hydrodynamic arrows, and you will see the exact gait your optimization algorithm discovered.

If the gait looks slightly unstable, try switching CHECKPOINT_FILE from "best_max.json" to "best_avg.json". 
The maximum rollout is sometimes a "lucky" run that survived a weird noise variation, 
while the average rollout represents a fundamentally solid walking pattern.
'''

# ======================================================
# CONFIGURATION
# ======================================================
CHECKPOINT_FILE = "data/checkpoints/best_avg.json" # Or best_max.json
SIMULATION_STEPS = 5000  # Run for 25 seconds (at 0.005s dt) to watch it walk

LEG_SIDE    = ['R', 'L']
LEG_INDEX   = ["F", "B"]
JOINT_NAMES = [0, 1, 2, 3]
NUM_KERNELS = 50 

if __name__ == "__main__":
    print(f"Loading checkpoint: {CHECKPOINT_FILE}")
    
    # 1. Load the trained parameters from the JSON file
    with open(CHECKPOINT_FILE, "r") as f:
        checkpoint_data = json.load(f)

    trained_weights = checkpoint_data["parameters"]
    print(f"Loaded weights from Iteration {checkpoint_data['iteration']} with Fitness: {checkpoint_data['fitness']:.4f}")

    # 2. Slice the weights for your joints (Exact same logic as training)
    imitated_weights = {}
    joint_keys = [(side, index, joint) for side in LEG_SIDE for index in LEG_INDEX for joint in JOINT_NAMES]
    
    for i, (side, index, joint) in enumerate(joint_keys):
        start_idx = i * NUM_KERNELS
        end_idx = start_idx + NUM_KERNELS
        dict_key = f'{index}{side}{joint}'
        imitated_weights[dict_key] = trained_weights[start_idx:end_idx]

    # 3. Initialize Neural Networks
    cpg = CPG_SO2()
    rbf = RBF()

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
            cpg_mod_cmd[f'{index}{side}']   = {'phi': 0.2, 'pause_input': 1.0, 'rewind_input': 1.0}

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
                
        env_targets = {}
        
        # Update RBFs to generate joint targets
        for side in LEG_SIDE:
            for index in LEG_INDEX:
                for joint in JOINT_NAMES:
                    target_angle = rbf.regenerate_target_traj(
                        cpg_output[f'{index}{side}']['cpg_output_0'], 
                        cpg_output[f'{index}{side}']['cpg_output_1'],
                        imitated_weights[f'{index}{side}{joint}']
                    )
                    
                    actuator_name = f"{index}{side}_{joint}" 
                    env_targets[actuator_name] = target_angle
            
        # Step the environment
        env.step(env_targets)
        
        # Optional: Add a tiny sleep to slow down the viewer to real-time 
        # if your computer calculates physics much faster than real-time
        # time.sleep(0.005) 

    print("Replay finished.")