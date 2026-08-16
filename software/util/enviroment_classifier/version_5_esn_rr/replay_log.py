import json
import time
import numpy as np
from script import StickInsectEnv
from cpg_rbf.cpg_so2 import CPG_SO2, CPG_LOCO
from cpg_rbf.rbf import RBF
from gait_cycle_cut.gait_cycle_cut import OnlineGaitSegmenter
import os
import subprocess
import signal
import datetime

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
TARGET_ITERATION = -1  # Set to -1 for the last iteration, or a specific number (e.g., 150)
SIMULATION_STEPS = 2000 # 12000 (1min.) 
CPG_PHI = 0.05

gait = "solid_ground"
environment_setup = "muddy_ground"

SAVE_ROS_BAG = False

LOG_FILE = f"learned_weight_set/{gait}/weight_set_3.json" 

SAVE_METRIC_CSV = False
MATRIC_FILE = f"terrain_dataset/metric/{gait}_gait_on_{environment_setup}"


SAVE_EXTRACTED_WEIGHTS = False 
EXTRACTED_WEIGHTS_FILE = f"learned_weight_set/{gait}/extracted_weights.npz" 

RECORD_TRAJECTORY = True  
recorded_trajectory = {}  # Will store lists of angles for each joint
EXTRACTED_TRAJ_FILE = f"learned_weight_set/{gait}/target_trajectory.json"


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
    save_to_file = SAVE_METRIC_CSV
    metric_filename = MATRIC_FILE

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

    

    # # ===============================================================
    # # RESIDUAL SYMMETRY LOGIC (Reconstructs Left offsets from Prior) (water surface)
    # # ===============================================================
    # print("Loading Prior Knowledge to reconstruct Left offsets...")
    # prior_knowledge = np.load('learned_weight_set/water_surface/imitated_diving_beetle_swim_forward_weights_20_kernels.npz')

    # imitated_weights = {}
    
    # joint_index = 0
    # for index in LEG_INDEX: 
    #     for joint in JOINT_NAMES:
    #         start_idx = joint_index * NUM_KERNELS
    #         end_idx = start_idx + NUM_KERNELS
            
    #         # 1. Get the Original Priors first so we can use them for the freeze
    #         right_key = f"{index}R{joint}"
    #         left_key = f"{index}L{joint}"
            
    #         right_prior = np.array(prior_knowledge[right_key])
    #         left_prior = np.array(prior_knowledge[left_key])
            
    #         # ==========================================
    #         # 🚨 THE FIX: FREEZE JOINT 1 
    #         # ==========================================
    #         # if joint == 1:
    #         #     # FREEZE: Ignore the JSON log. Force it to be the original prior.
    #         #     learned_weights = np.copy(right_prior)
    #         # else:
    #         #     # LEARN: Get the optimized weights from the JSON log
    #         #     learned_weights = np.array(trained_weights[start_idx:end_idx])

    #         learned_weights = np.array(trained_weights[start_idx:end_idx])
            
    #         # Calculate the exact difference (Left - Right)
    #         offset_weights = left_prior - right_prior
            
    #         # 3. Assign the pure weights to the RIGHT side
    #         imitated_weights[right_key] = learned_weights
            
    #         # 4. Assign the Weights + Original Offset to the LEFT side
    #         imitated_weights[left_key] = learned_weights + offset_weights
            
    #         joint_index += 1
    # # ===============================================================

    # # ===============================================================
    # # RESIDUAL SYMMETRY LOGIC (Reconstructs Left offsets from Prior)
    # # ===============================================================
    # print("Loading Prior Knowledge to reconstruct Left offsets...")
    # prior_knowledge = np.load('imitated_diving_beetle_swim_forward_weights_20_kernels.npz')

    # imitated_weights = {}
    
    # joint_index = 0
    # for index in LEG_INDEX: 
    #     for joint in JOINT_NAMES:
    #         start_idx = joint_index * NUM_KERNELS
    #         end_idx = start_idx + NUM_KERNELS
            
    #         # 1. Get the Learned Weights (for the Right side) from the JSON log
    #         learned_weights = np.array(trained_weights[start_idx:end_idx])
            
    #         # 2. Get the Original Priors to calculate the offset
    #         right_key = f"{index}R{joint}"
    #         left_key = f"{index}L{joint}"
            
    #         right_prior = np.array(prior_knowledge[right_key])
    #         left_prior = np.array(prior_knowledge[left_key])
            
    #         # Calculate the exact difference (Left - Right)
    #         offset_weights = left_prior - right_prior
            
    #         # 3. Assign the pure learned weights to the RIGHT side
    #         imitated_weights[right_key] = learned_weights
            
    #         # 4. Assign the Learned Weights + Original Offset to the LEFT side
    #         imitated_weights[left_key] = learned_weights
            
    #         joint_index += 1
    # # =========================================================== ====

    # ===============================================================
    # WEIGHT SYMMETRY LOGIC (walking)
    # ===============================================================
    imitated_weights = {}
    
    # 1. We only loop through the Front ('F') and Back ('B') indices
    joint_index = 0
    for index in LEG_INDEX: 
        for joint in JOINT_NAMES:
            start_idx = joint_index * NUM_KERNELS
            end_idx = start_idx + NUM_KERNELS
            
            # Extract the 20 weights for this specific joint
            extracted_weights = trained_weights[start_idx:end_idx]
            
            # 2. Assign these weights to the RIGHT side
            right_key = f"{index}R{joint}"
            imitated_weights[right_key] = extracted_weights
            
            # 3. MIRROR them exactly to the LEFT side!
            left_key = f"{index}L{joint}"
            imitated_weights[left_key] = extracted_weights
            
            joint_index += 1
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

    # imitated_weights = {k: np.array(v) for k, v in imitated_weights.items()}
    # # ===============================================================
    # # STEP 1: BAKE INVERSIONS & OFFSETS (DO THIS FIRST!)
    # # ===============================================================
    # for index in LEG_INDEX: # Loops through ['F', 'B']
        
    #     # 1. Left hip (J0) inversion is always applied
    #     key_L0 = f"{index}L0"
    #     if key_L0 in imitated_weights:
    #         imitated_weights[key_L0] = -imitated_weights[key_L0]
            
    #     # 2. Left J1 inversion depends on the gait
    #     key_L1 = f"{index}L1"
    #     if key_L1 in imitated_weights:
    #         if "ground" in gait:
    #             # For walking: network_output = -network_output
    #             imitated_weights[key_L1] = -imitated_weights[key_L1]
    #             print(f"Applied walking gait")
                
    #         elif "water" in gait:
    #             # For swimming: network_output = -network_output - np.pi
    #             imitated_weights[key_L1] = -imitated_weights[key_L1]
    #             print(f"Applied swimming gait")


    # ===============================================================
    # STEP 2: BAKE STANDING POSE / BASELINE (DO THIS SECOND!)
    # ===============================================================
    # for side in LEG_SIDE:       # ['R', 'L']
    #     for index in LEG_INDEX: # ['F', 'B']
    #         for joint in JOINT_NAMES: # [0, 1, 2, 3]
                
    #             key = f"{index}{side}{joint}"
    #             if "ground" in gait:    
    #                 if key in imitated_weights:
    #                     # 1. Fetch the baseline angle from your dictionary
    #                     baseline_angle = STANDING_POSE[f'{index}{side}'][joint]
                        
    #                     # 2. Add the baseline angle to the weights
    #                     imitated_weights[key] = imitated_weights[key] + baseline_angle

    # ==========================================
    # SAVE THE SELECTED WEIGHTS AS JSON
    # ==========================================
    if SAVE_EXTRACTED_WEIGHTS:
        import os
        import json
        
        json_filename = f"learned_weight_set/{gait}/extracted_weights.json"
        os.makedirs(os.path.dirname(json_filename), exist_ok=True)
        
        # Convert all numpy arrays in the dictionary to lists
        json_ready_weights = {key: value.tolist() for key, value in imitated_weights.items()}
        
        with open(json_filename, "w") as f:
            json.dump(json_ready_weights, f, indent=4)
            
        print(f"✅ Reconstructed weights saved to JSON: {json_filename}")
    # ==========================================


    # ==========================================
    # SETUP ONLINE SEGMENTER
    # ==========================================
    segmenter = OnlineGaitSegmenter()
    latest_complete_cycle = None
    
    # Create a fixed order of joint names so we can map them back later
    ordered_actuator_names = []
    for side in LEG_SIDE:
        for index in LEG_INDEX:
            for joint in JOINT_NAMES:
                ordered_actuator_names.append(f"{index}{side}_J{joint+1}")

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
    

    # ==========================================
    # START ROS 2 BAG RECORDING
    # ==========================================
    if SAVE_ROS_BAG:
        bag_process = None
        if env.enable_ros:
            # Create a unique bag name to prevent ROS crashes if the folder already exists
            timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            bag_dir = f"classification_model_training/rosbags/{gait}_gait_on_{environment_setup}"
            
            # Ensure the parent directory exists
            os.makedirs(os.path.dirname(bag_dir), exist_ok=True)
            
            print(f"🎬 Starting ROS 2 bag recording: {bag_dir}")
            
            # Launch the recording in the background (-a records all topics)
            # We suppress stdout/stderr so it doesn't spam your terminal
            bag_process = subprocess.Popen(
                ["ros2", "bag", "record", "-a", "-o", bag_dir],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
    # ==========================================

    print("Starting replay...")
    
    # 5. ==================================================================================== Run the Simulation Loop ==============================================================================================================================
    for step in range(SIMULATION_STEPS):
        step_start = time.perf_counter() # use perf_counter for high precision
        # Update CPGs
        for side in LEG_SIDE:
            for index in LEG_INDEX:
                cpg_output[f'{index}{side}'] = cpg_modulated[f'{index}{side}'].modulate_cpg(
                    cpg_mod_cmd[f'{index}{side}']['phi'], 
                    cpg_mod_cmd[f'{index}{side}']['pause_input'], 
                    cpg_mod_cmd[f'{index}{side}']['rewind_input']
                )
        
        # Create a FLAT dictionary for the environment targets (e.g. {'FR_J1': 0.1, 'FR_J2': 0.5})
        joint_targets = {}
        cpg_outputs = {}
        
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
                    
                    # # ==========================================
                    # # NEW: MECHANICAL INVERSION FOR LEFT HIP
                    # # ==========================================
                    # # Because the Left hip (J0) axis is mirrored in the XML, 
                    # # we must invert the network output to make it swing the 
                    # # same physical direction as the Right hip.
                    if side == 'L' and joint == 0:
                        network_output = -network_output 

                    # # for walking
                    if side == 'L' and joint == 1:
                        network_output = -network_output 
                    # # for swimming
                    # if side == 'L' and joint == 1:
                    #     network_output = -network_output - np.pi
                    
                    # 2. Add it to your standing pose
                    baseline_angle = STANDING_POSE[f'{index}{side}'][joint]
                    target_angle =  network_output + baseline_angle
                    # target_angle =  network_output
                    
                    # 3. Store it using the exact string format your MuJoCo XML actuators use
                    actuator_name = f"{index}{side}_J{joint+1}"  
                    joint_targets[actuator_name] = target_angle
                    cpg_outputs[actuator_name] = cpg_output[f'{index}{side}']['cpg_output_0']
                    
                    # ==========================================
                    # RECORD THE FINAL ANGLE
                    # ==========================================
                    if RECORD_TRAJECTORY:
                        if actuator_name not in recorded_trajectory:
                            recorded_trajectory[actuator_name] = []
                        # Cast to standard python float() to avoid JSON serialization errors with numpy floats
                        recorded_trajectory[actuator_name].append(float(target_angle))
            
        # Pass the flat dictionary to the environment
        env.step(joint_targets, cpg_outputs)

        # ==========================================
        # FEED DATA TO SEGMENTER
        # ==========================================
        if RECORD_TRAJECTORY:
            # 1. Grab all 16 target angles in the exact order we defined above
            current_joint_angles = [joint_targets[name] for name in ordered_actuator_names]
            
            # 2. Use one of the CPG outputs as our "clock" (FR leg is standard)
            reference_cpg = cpg_output['FR']['cpg_output_0']
            
            # 3. Add to segmenter
            cycle = segmenter.add_data_point(reference_cpg, current_joint_angles)
            
            # 4. If a cycle just finished, store it! 
            if cycle is not None:
                latest_complete_cycle = cycle
        # ==========================================

        target_duration = env.model.opt.timestep
        while (time.perf_counter() - step_start) < target_duration:
            time.sleep(0.0001) # Yields the CPU instead of locking it
    # ========================================================================================================================================================================
    print("Replay finished.")
    
    # ==========================================
    # SAVE THE RECORDED TRAJECTORY TO JSON
    # ==========================================
    if RECORD_TRAJECTORY:
        # ==========================================
        # NORMALIZE AND SAVE THE LAST CYCLE TO JSON
        # ==========================================
        if latest_complete_cycle is not None:
            print(f"Normalizing the last captured cycle to {cpg_cycle_length} points...")
            
            # 1. Normalize the cycle to perfectly match your CPG cycle length
            normalized_cycle = segmenter.normalize_cycle(
                latest_complete_cycle, 
                num_points=cpg_cycle_length
            )
            
            # 2. Convert the 2D numpy array back into a dictionary using our ordered names
            final_trajectory_dict = {}
            for idx, name in enumerate(ordered_actuator_names):
                # Convert to standard Python lists for JSON saving
                final_trajectory_dict[name] = normalized_cycle[:, idx].tolist()
                
            # 3. Save to JSON
            import os
            import json
            
            traj_filename = EXTRACTED_TRAJ_FILE
            os.makedirs(os.path.dirname(traj_filename), exist_ok=True)
            
            with open(traj_filename, "w") as f:
                json.dump(final_trajectory_dict, f, indent=4)
                
            print(f"✅ One perfect gait cycle successfully saved to: {traj_filename}")
        else:
            print("⚠️ Warning: No complete cycles were detected during the simulation.")
    # ==========================================
    # STOP ROS 2 BAG RECORDING
    # ==========================================
    if SAVE_ROS_BAG:
        if bag_process is not None:
            print("🛑 Stopping ROS 2 bag recording and saving file...")
            # Send SIGINT (Ctrl+C) so ROS 2 properly closes the sqlite3 database
            bag_process.send_signal(signal.SIGINT)
            bag_process.wait() # Wait for the file to finish saving
            print("✅ ROS bag successfully saved!")
    # ==========================================
    env.close()
    print("Simulation safely closed.")

    # metric = env.calculate_gait_metrics()
    # if save_to_file:
    #     import os
    #     import csv
    #     import datetime
        
    #     # Ensure the filename ends with .csv
    #     if not metric_filename.endswith('.csv'):
    #         metric_filename += '.csv'
            
    #     # Create the directory if it doesn't exist yet! 
    #     # (e.g., 'terrain_dataset/metric/')
    #     os.makedirs(os.path.dirname(metric_filename), exist_ok=True)
        
    #     timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    #     file_exists = os.path.isfile(metric_filename)
        
    #     # Use 'a' to append. newline="" prevents blank rows in Windows.
    #     with open(metric_filename, mode="a", newline="") as f:
    #         # Define our columns. (Added Fitness_Score since this is a replay!)
    #         fieldnames = ["Timestamp", "Total_Steps", "Fitness_Score"] + list(metric.keys())
    #         writer = csv.DictWriter(f, fieldnames=fieldnames)
            
    #         # Write the header row ONLY if the file is brand new
    #         if not file_exists:
    #             writer.writeheader()
            
    #         # Prepare the row data
    #         row_data = {
    #             "Timestamp": timestamp, 
    #             "Total_Steps": env.step_count,           # Fixed: Changed from self to env
    #             "Fitness_Score": f"{fitness_score:.4f}"  # Logs the fitness from the JSON
    #         }
            
    #         # Format numbers to 4 decimal places and add to row_data
    #         for k, v in metric.items():                  # Fixed: Changed metrics to metric
    #             row_data[k] = f"{v:.6f}"
                
    #         # Write the row to the CSV
    #         writer.writerow(row_data)
            
    #     print(f"Metrics successfully saved to: {metric_filename}")

    

   

    