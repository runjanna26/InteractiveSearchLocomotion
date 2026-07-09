import json
import os
import glob
import numpy as np
import matplotlib.pyplot as plt

# Import your neural network classes exactly as they are in your main script
from cpg_rbf.cpg_so2 import CPG_SO2
from cpg_rbf.rbf import RBF

# ==========================================
# CONFIGURATION
# ==========================================
LOG_DIR = "data/pibb_logs/"
SPECIFIC_LOG_FILE = "data/pibb_logs/pibb_training_20260708_155502.json" 

# Which iterations do you want to compare on the graph?
# Pick a spread: e.g., the start, a middle point, and the final result.
ITERATIONS_TO_PLOT = [0, 50, 150, 349] 

# Network constraints (Must match main_learning.py)
NUM_KERNELS = 20
CPG_PHI = 0.02
LEG_SIDE    = ['R', 'L']
LEG_INDEX   = ["F", "B"]
JOINT_NAMES = [0, 1, 2, 3]

def get_latest_log_file(directory):
    list_of_files = glob.glob(os.path.join(directory, '*.json'))
    if not list_of_files:
        return None
    return max(list_of_files, key=os.path.getctime)

def extract_joint_weights(flat_parameters, target_joint):
    """
    Finds the exact slice of 20 weights for the requested joint 
    from the massive flat list of parameters.
    """
    joint_keys = [f'{index}{side}{joint}' for side in LEG_SIDE for index in LEG_INDEX for joint in JOINT_NAMES]
    
    try:
        joint_index = joint_keys.index(target_joint)
    except ValueError:
        print(f"Error: Joint {target_joint} not found. Valid options: {joint_keys}")
        exit()
        
    start_idx = joint_index * NUM_KERNELS
    end_idx = start_idx + NUM_KERNELS
    return flat_parameters[start_idx:end_idx]

def plot_learned_trajectories():
    log_file = SPECIFIC_LOG_FILE if SPECIFIC_LOG_FILE else get_latest_log_file(LOG_DIR)
    
    if not log_file:
        print(f"Error: No JSON logs found in {LOG_DIR}")
        return
        
    print(f"Loading data from: {log_file}")
    with open(log_file, 'r') as f:
        log_data = json.load(f)

    # 1. Initialize the CPG and RBF networks to generate the base cycle
    cpg = CPG_SO2()
    rbf = RBF(nc=NUM_KERNELS)

    cpg_one_cycle = cpg.generate_cpg_one_cycle(CPG_PHI)       
    out0 = cpg_one_cycle['out0_cpg_one_cycle'][:]
    out1 = cpg_one_cycle['out1_cpg_one_cycle'][:]
    cpg_cycle_length = len(out0)
    
    rbf.construct_kernels_with_cpg_one_cycle(out0, out1, cpg_cycle_length)
    
    # Create a time array (X-axis) representing one full step cycle
    time_steps = np.arange(cpg_cycle_length)

    # Create a 4x4 grid of subplots for all 16 joints
    fig, axs = plt.subplots(4, 4, figsize=(20, 16), sharex=True, sharey=True)
    fig.suptitle('Evolution of Rhythmic Trajectories for All Joints', fontsize=22, fontweight='bold')

    # To keep plots organized: Rows = Legs, Cols = Joints
    # LEG_SIDE = ['R', 'L'], LEG_INDEX = ['F', 'B'] -> ['FR', 'BR', 'FL', 'BL']
    legs = [f"{index}{side}" for side in LEG_SIDE for index in LEG_INDEX] 
    
    for i, leg in enumerate(legs):
        for j, joint in enumerate(JOINT_NAMES):
            target_joint = f"{leg}{joint}"
            ax = axs[i, j]

            # Extract and Plot the trajectory for each requested iteration
            for target_iter in ITERATIONS_TO_PLOT:
                # Find the log entry for this iteration
                entry = next((item for item in log_data if item["iteration"] == target_iter), None)
                
                if entry is None:
                    continue
                    
                # Extract the best parameters found in that iteration
                best_flat_params = entry["best_parameters"]
                joint_weights = extract_joint_weights(best_flat_params, target_joint)
                
                # Generate the rhythmic target angle for every step in the CPG cycle
                trajectory = []
                for t in range(cpg_cycle_length):
                    angle = rbf.regenerate_target_traj(out0[t], out1[t], joint_weights)
                    trajectory.append(angle)
                    
                # Plot it
                ax.plot(time_steps, trajectory, label=f'Iter {target_iter}', linewidth=2)

            # Subplot formatting
            ax.set_title(f'Joint: {target_joint}', fontsize=14, fontweight='bold')
            ax.axhline(0, color='black', linewidth=1, linestyle='--') 
            ax.grid(True, linestyle=':', alpha=0.7)
            
            # Only show x-labels on the bottom row and y-labels on the left col to avoid clutter
            if i == 3:
                ax.set_xlabel('CPG Phase (Time Steps)', fontsize=12)
            if j == 0:
                ax.set_ylabel('Angle Offset (Rad)', fontsize=12)

    # Add a single legend for the entire figure to save space in the individual subplots
    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', fontsize=14, bbox_to_anchor=(0.98, 0.98))
    
    plt.tight_layout()
    # Adjust top boundary to make room for the master title and legend
    plt.subplots_adjust(top=0.92)
    
    # Save the plot
    save_path = "trajectory_all_joints.png"
    plt.savefig(save_path, dpi=300)
    print(f"Saved plot to: {save_path}")
    
    plt.show()

if __name__ == "__main__":
    plot_learned_trajectories()