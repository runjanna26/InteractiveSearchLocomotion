import json
import time
import numpy as np
import os
import subprocess
import signal
import datetime

from script import StickInsectEnv
from cpg_rbf.cpg_so2 import CPG_SO2, CPG_LOCO
from cpg_rbf.rbf import RBF

# ======================================================
# CONFIGURATION
# ======================================================
gait = "water_surface"
environment_setup = "solid_ground"

# Point directly to your newly learned JSON weights
WEIGHTS_FILE = f"learned_weight_set/{gait}/learned_weights.json"

SIMULATION_STEPS = 2000 # 12000 (1min.) 
CPG_PHI = 0.05

SAVE_ROS_BAG = False
SAVE_METRIC_CSV = False
MATRIC_FILE = f"terrain_dataset/metric/{gait}_gait_on_{environment_setup}"

LEG_SIDE    = ['R', 'L']
LEG_INDEX   = ["F", "B"]
JOINT_NAMES = [0, 1, 2, 3]
NUM_KERNELS = 20 

if __name__ == "__main__":
    print(f"Loading learned weights from: {WEIGHTS_FILE}")
    
    # ==========================================
    # 1. LOAD THE LEARNED WEIGHTS
    # ==========================================
    try:
        with open(WEIGHTS_FILE, "r") as f:
            raw_learned_weights = json.load(f)
    except FileNotFoundError:
        print(f"Error: Could not find '{WEIGHTS_FILE}'. Please run the imitation learning script first.")
        exit()

    # Convert all lists back into NumPy arrays for mathematical operations
    learned_weights = {k: np.array(v) for k, v in raw_learned_weights.items()}

    # ==========================================
    # 2. INITIALIZE NEURAL NETWORKS
    # ==========================================
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

    # half_cycle = cpg_cycle_length // 2
    
    # for _ in range(half_cycle):
    #     # Manually step the FL and BR oscillators forward in time
    #     # before the MuJoCo simulation even begins to create the trot pattern.
    #     cpg_output['FL'] = cpg_modulated['FL'].modulate_cpg(CPG_PHI, 0.0, 1.0)
    #     cpg_output['BR'] = cpg_modulated['BR'].modulate_cpg(CPG_PHI, 0.0, 1.0)

    # ==========================================
    # 3. INITIALIZE ENVIRONMENT
    # ==========================================
    env = StickInsectEnv(enable_ros=True, render=True) 
    env.reset()
    
    # ==========================================
    # START ROS 2 BAG RECORDING (Optional)
    # ==========================================
    bag_process = None
    if SAVE_ROS_BAG and env.enable_ros:
        timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        bag_dir = f"classification_model_training/rosbags/{gait}_gait_on_{environment_setup}"
        
        os.makedirs(os.path.dirname(bag_dir), exist_ok=True)
        print(f"🎬 Starting ROS 2 bag recording: {bag_dir}")
        
        bag_process = subprocess.Popen(
            ["ros2", "bag", "record", "-a", "-o", bag_dir],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )

    print("Starting replay with mathematically imitated weights...")
    
    # ==========================================
    # 4. SIMULATION LOOP
    # ==========================================
    for step in range(SIMULATION_STEPS):
        step_start = time.perf_counter() 
        
        # Update CPGs
        for side in LEG_SIDE:
            for index in LEG_INDEX:
                cpg_output[f'{index}{side}'] = cpg_modulated[f'{index}{side}'].modulate_cpg(
                    cpg_mod_cmd[f'{index}{side}']['phi'], 
                    cpg_mod_cmd[f'{index}{side}']['pause_input'], 
                    cpg_mod_cmd[f'{index}{side}']['rewind_input']
                )
        
        joint_targets = {}
        cpg_outputs = {}
        
        # Update RBFs
        for side in LEG_SIDE:
            for index in LEG_INDEX:
                for joint in JOINT_NAMES:  
                    
                    # Look up the specific weight array (e.g., 'FR0')
                    weight_key = f"{index}{side}{joint}"
                    weights = learned_weights[weight_key]
                    
                    # Calculate the target angle directly from the neural network
                    # (Inversions, offsets, and standing poses are already baked into these weights!)
                    target_angle = rbf.regenerate_target_traj(
                        cpg_output[f'{index}{side}']['cpg_output_0'], 
                        cpg_output[f'{index}{side}']['cpg_output_1'],
                        weights
                    )
                    
                    actuator_name = f"{index}{side}_J{joint+1}"  
                    joint_targets[actuator_name] = target_angle
                    cpg_outputs[actuator_name] = cpg_output[f'{index}{side}']['cpg_output_0']
            
        # Pass the flat dictionary to the environment
        env.step(joint_targets, cpg_outputs)

        target_duration = env.model.opt.timestep
        while (time.perf_counter() - step_start) < target_duration:
            pass # Busy-wait ensures perfect timing

    print("Replay finished.")
    
    # ==========================================
    # STOP ROS 2 BAG RECORDING
    # ==========================================
    if SAVE_ROS_BAG:
        if bag_process is not None:
            print("🛑 Stopping ROS 2 bag recording and saving file...")
            bag_process.send_signal(signal.SIGINT)
            bag_process.wait() 
            print("✅ ROS bag successfully saved!")

    # ==========================================
    # SAVE METRICS (Optional)
    # ==========================================
    if SAVE_METRIC_CSV:
        metric = env.calculate_gait_metrics()
        import csv
        
        if not MATRIC_FILE.endswith('.csv'):
            MATRIC_FILE += '.csv'
            
        os.makedirs(os.path.dirname(MATRIC_FILE), exist_ok=True)
        
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        file_exists = os.path.isfile(MATRIC_FILE)
        
        with open(MATRIC_FILE, mode="a", newline="") as f:
            fieldnames = ["Timestamp", "Total_Steps", "Imitation_Status"] + list(metric.keys())
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            
            if not file_exists:
                writer.writeheader()
            
            row_data = {
                "Timestamp": timestamp, 
                "Total_Steps": env.step_count,
                "Imitation_Status": "Learned_RBF_Weights"
            }
            
            for k, v in metric.items():
                row_data[k] = f"{v:.6f}"
                
            writer.writerow(row_data)
            
        print(f"Metrics successfully saved to: {MATRIC_FILE}")