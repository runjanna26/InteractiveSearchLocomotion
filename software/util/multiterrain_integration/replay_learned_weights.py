import json
import time
import numpy as np
import os
import subprocess
import signal
import datetime
import itertools
import random
import threading
import queue

from script import StickInsectEnv
from cpg_rbf.cpg_so2 import CPG_SO2, CPG_LOCO
from cpg_rbf.rbf import RBF
from gait_cycle_cut.gait_cycle_cut import OnlineGaitSegmenter
from classification_model_training.models.model_1_esn_rr.env_pred import ESN_RR_Classification

# ======================================================
# CONFIGURATION
# ======================================================

# 'solid_ground', 
# 'soft_ground', 
# 'slippery_ground', 
# 'muddy_ground', 
# 'water_surface'

gait = "water_surface"
environment_setup = "water_surface"

ENABLE_VIDEO_REC = False

ENABLE_ROS = True
SAVE_ROS_BAG = False
ROS_BAG_FILE = f"final_transition/attemp_15"
# ROS_BAG_FILE = f"classification_model_training/dataset_for_esn/{gait}_gait_on_{environment_setup}_set_2"

SAVE_METRIC_CSV = False
METRIC_FILE = f"metric/{gait}_gait_on_{environment_setup}"

ENABLE_RENDERING = True
ENABLE_ESN = True

LEG_SIDE    = ['R', 'L']
LEG_INDEX   = ["F", "B"]
JOINT_NAMES = [0, 1, 2, 3]
NUM_KERNELS = 20 

CPG_PHI = 0.05 
SIM_DURATION = 160.0 # seconds

# ======================================================
# 1. HELPER: LOAD JSON WEIGHTS
# ======================================================
def load_json_weights(filepath):
    try:
        with open(filepath, 'r') as f:
            raw_weights = json.load(f)
        return {k: np.array(v) for k, v in raw_weights.items()}
    except FileNotFoundError:
        print(f"Error: {filepath} not found. Please ensure the file exists.")
        exit()

def flatten_datasets(dataset_dict, terrain_to_id_map, sensor_order):
	"""
	Converts the nested dictionary into flat X and y lists for the ESN.
	
	dataset_dict: The output from your split_dataset function.
	terrain_to_id_map: Dictionary mapping string terrain names to integers (e.g., {'concrete': 0, 'grass': 1}).
	sensor_order: A strict list of sensor dictionary keys to ensure columns are always in the same order.
	"""
	X_formatted = []
	y_formatted = []
	
	for terrain, sensors_data in dataset_dict.items():
		if terrain not in terrain_to_id_map:
			continue # Skip terrains we aren't training on
			
		label = terrain_to_id_map[terrain]
		
		# Get the number of cycles for this terrain (they are synchronized)
		n_cycles = len(sensors_data[sensor_order[0]])
		
		for i in range(n_cycles):
			# For cycle 'i', gather data from all 16 joints (K, D, tau)
			cycle_columns = []
			for sensor_name in sensor_order:
				# Ensure it's a numpy array
				sensor_array = np.array(sensors_data[sensor_name][i])
				
				# If the sensor data is 1D (e.g., shape (90,)), reshape to (90, 1) so it stacks horizontally
				if len(sensor_array.shape) == 1:
					sensor_array = sensor_array.reshape(-1, 1)
					
				cycle_columns.append(sensor_array)
			
			# Stack all sensors horizontally to create the (90 timesteps x 48 features) matrix
			full_gait_matrix = np.hstack(cycle_columns)
			
			X_formatted.append(full_gait_matrix)
			y_formatted.append(label)
			
	return X_formatted, np.array(y_formatted)

if __name__ == "__main__":
    print("Loading multiple learned weight sets for adaptive transition...")
    
    # Load all four environments
    weights_solid = load_json_weights("learned_weight_set/solid_ground/learned_weights.json")
    weights_water = load_json_weights("learned_weight_set/water_surface/learned_weights.json")
    weights_soft  = load_json_weights("learned_weight_set/soft_ground/learned_weights.json")
    weights_rough = load_json_weights("learned_weight_set/rough_ground/learned_weights.json")
    weights_slip  = load_json_weights("learned_weight_set/slippery_ground/learned_weights.json")
    weights_muddy = load_json_weights("learned_weight_set/muddy_ground/learned_weights.json")

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

    cpg_modulated = {}
    cpg_output = {}
    cpg_mod_cmd = {}

    for side in LEG_SIDE:
        for index in LEG_INDEX:
            cpg_modulated[f'{index}{side}'] = CPG_LOCO()
            cpg_output[f'{index}{side}']    = cpg_modulated[f'{index}{side}'].modulate_cpg(CPG_PHI, 0.0, 1.0)
            cpg_mod_cmd[f'{index}{side}']   = {'phi': CPG_PHI, 'pause_input': 0.0, 'rewind_input': 1.0}

    # ==========================================
    # 3. INITIALIZE SMOOTHING STATE (For all 16 joints)
    # ==========================================
    rbf_weight_prev = {}
    for side in LEG_SIDE:
        for index in LEG_INDEX:
            for joint in JOINT_NAMES:
                key = f"{index}{side}{joint}"
                # Initialize all joints to start with the Solid Ground gait
                rbf_weight_prev[key] = np.copy(weights_solid[key])

    # ==========================================
    # 3.5 INITIALIZE ONLINE SEGMENTERS & ESN
    # ==========================================
    # Define the exact sensors your ESN Config 8 expects
    sensors_to_segment = [
        'cpg_output',
        'joint_commands',
        'joint_stiffness_fb', 
        'joint_damping_fb', 
        'joint_torque_feedforward_fb', 
        'foot_force', 
        'joint_angle_fb', 
        'joint_velocity_fb'
    ]
    
    # Create one independent segmenter for each sensor array
    segmenters = {sensor: OnlineGaitSegmenter() for sensor in sensors_to_segment}
    
    # Load ESN Model
    environment_classifier_model = ESN_RR_Classification()
    environment_classifier_model.load_model("classification_model_training/trained_models_final/model_1_esn_config_11_w_cma_es.pt")
    
    all_probabilities = []
    
    # Terrain Mapping Dictionary for printing the prediction
    ID_TO_TERRAIN = {0: 'solid_ground',
                    1: 'slippery_ground',
                    2: 'muddy_ground',
                    3: 'water_surface'}
    
    # Map terrain IDs directly to their loaded weights for dynamic switching
    ID_TO_WEIGHTS = {
        0: weights_solid,
        1: weights_slip,
        2: weights_muddy,
        3: weights_water
    }
        
    # ==========================================
    # 4. THREADING ARCHITECTURE SETUP
    # ==========================================
    cycle_queue = queue.Queue(maxsize=5)
    state_lock = threading.Lock()
    
    # This dictionary securely shares state between the classifier thread and physics loop
    shared_state = {
        'active_weights_target': weights_solid,
        'trigger_slowdown': False,
        'all_probabilities': []
    }
    
    REQUIRED_CONSECUTIVE_PREDS = 3
    
    def classifier_worker():
        current_gait_class = 0          
        last_predicted_class = -1       
        consecutive_pred_count = 0      
        
        while True:
            item = cycle_queue.get()
            if item is None:  # Exit signal
                break
                
            normalized_cycle_dict, sim_time = item
            
            # --- Reconstruct the 2D Matrix (NO DUMMY TIME COLUMN) ---
            time_col = np.linspace(0, 1, 100).reshape(-1, 1) 
            
            esn_input_list = []
            for s in sensors_to_segment:
                sensor_data = normalized_cycle_dict[s]
                if len(sensor_data.shape) == 1:
                    sensor_data = sensor_data.reshape(-1, 1)
                
                # Re-add the time column to match the offline dataset's shape (124 columns total)
                sensor_w_time = np.hstack([sensor_data, time_col])
                esn_input_list.append(sensor_w_time)
                
            esn_input = np.concatenate(esn_input_list, axis=1)
            
            # --- Inference ---
            predicted_class, confidence, all_probabilities = environment_classifier_model.predict(esn_input)
            predicted_terrain_name = ID_TO_TERRAIN.get(predicted_class, "Unknown Terrain")
            
            print(f"\n🤖 [Time: {sim_time:.2f}s] ESN Predicts: {predicted_terrain_name.upper()} (Class {predicted_class}) with {confidence:.2f}% confidence.")
            
            # --- Debounce & Reflex Logic ---
            trigger_slowdown = False
            new_weights = None
            
            if predicted_class == last_predicted_class:
                consecutive_pred_count += 1
            else:
                consecutive_pred_count = 1
                last_predicted_class = predicted_class
                
                # Instant cautious reflex on terrain boundary
                if predicted_class != current_gait_class:
                    trigger_slowdown = True
                    
            print(f"   [Prediction Counter: {consecutive_pred_count} / {REQUIRED_CONSECUTIVE_PREDS}]")

            if consecutive_pred_count >= REQUIRED_CONSECUTIVE_PREDS and confidence >= 0.1:
                if predicted_class in ID_TO_WEIGHTS and current_gait_class != predicted_class:
                    new_weights = ID_TO_WEIGHTS[predicted_class]
                    current_gait_class = predicted_class
                    print(f"🔄 Consistent predictions reached! Adapting gait target to: {predicted_terrain_name.upper()}...")
            
            # --- Update Shared State Safely ---
            with state_lock:
                shared_state['all_probabilities'] = all_probabilities
                if trigger_slowdown:
                    shared_state['trigger_slowdown'] = True
                if new_weights is not None:
                    shared_state['active_weights_target'] = new_weights
                    
            cycle_queue.task_done()

    # Start the background thread
    if ENABLE_ESN:
        classifier_thread = threading.Thread(target=classifier_worker, daemon=True)
        classifier_thread.start()
    # ==========================================
    # 4. INITIALIZE ENVIRONMENT
    # ==========================================
    env = StickInsectEnv(enable_ros=ENABLE_ROS, render=ENABLE_RENDERING, record_video=ENABLE_VIDEO_REC) 
    env.reset()
    
    # Calculate exact steps needed for 20 seconds based on MuJoCo timestep
    dt = env.model.opt.timestep
    total_steps = int(SIM_DURATION / dt)
    
    # ==========================================
    # START ROS 2 BAG RECORDING (Optional)
    # ==========================================
    bag_process = None
    if SAVE_ROS_BAG and env.enable_ros:
        timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        bag_dir = ROS_BAG_FILE
        
        os.makedirs(os.path.dirname(bag_dir), exist_ok=True)
        print(f"🎬 Starting ROS 2 bag recording: {bag_dir}")
        
        bag_process = subprocess.Popen(
            ["ros2", "bag", "record", "-a", "-o", bag_dir],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )

    # Set the initial default gait (e.g., solid ground) before the loop starts
    active_weights_target = weights_solid
    
    # --- DEBOUNCE / ROBUSTNESS TRACKERS ---
    current_gait_class = 0          
    last_predicted_class = -1       
    consecutive_pred_count = 0      

    # --- NOVEL CAUTIOUS STEPPING REFLEX ---
    current_phi = CPG_PHI           # Track the live speed
    
    # phi_recovery_rate = 0.0000075     
    phi_recovery_rate = 0.000005     
    # phi_recovery_rate = 0.00005     
    slow_phi_target = CPG_PHI * 0.5 # Drops speed to 50% during transition
    # --------------------------------------
    
    # --------------------------------------
    TRANSTION_RATE = 0.0025
    # --------------------------------------
    
    print("Starting adaptive simulation with smooth weight transitions...")
    
    base_gaits = [
        "solid_ground", 
        "muddy_ground", 
        "slippery_ground", 
        "water_surface", 
        "soft_ground"
    ]
    
    # 1. Generate every possible 2-gait transition (20 total pairs)
    all_pairs = list(itertools.permutations(base_gaits, 2))
    
    # 2. Shuffle the pairs so the transitions happen in a random order
    random.shuffle(all_pairs)
    
    # 3. Flatten the pairs into a single continuous sequence
    # This creates a list of 40 gaits (20 pairs * 2 gaits per pair)
    gait_sequence = []
    for pair in all_pairs:
        gait_sequence.extend(pair)
    
    gait_phase = 0
    switch_interval = 10.0  # Switch gait every 10 seconds
    
    print("Starting adaptive simulation with smooth weight transitions...")
    
    # ==========================================
    # 5. SIMULATION LOOP
    # ==========================================
    for step in range(total_steps):
        step_start = time.perf_counter() 
        sim_time = step * dt
        
        # --------------------------------------------------
        # GAIT TARGET SELECTION BASED ON TIME
        # --------------------------------------------------
        
        # if gait == "solid_ground":
        #     active_weights_target = weights_solid
        # elif gait == "water_surface":
        #     active_weights_target = weights_water
        # elif gait == "soft_ground":
        #     active_weights_target = weights_soft
        # elif gait == "rough_ground":
        #     active_weights_target = weights_rough
        # elif gait == "slippery_ground":
        #     active_weights_target = weights_slip
        # elif gait == "muddy_ground":
        #     active_weights_target = weights_muddy
        # else:
        #     print(f"Error: Unknown gait '{gait}'. Please choose a valid gait.")
        #     exit()
        
        # --------------------------------------------------
        # GAIT TARGET SELECTION BASED ON TIME
        # --------------------------------------------------
        # if step % int(switch_interval / dt) == 0:
            
        #     # Ensure we haven't run out of combinations
        #     if gait_phase < len(gait_sequence):
        #         current_gait = gait_sequence[gait_phase]
                
        #         if current_gait == "solid_ground":
        #             active_weights_target = weights_solid
        #         elif current_gait == "water_surface":
        #             active_weights_target = weights_water
        #         elif current_gait == "soft_ground":
        #             active_weights_target = weights_soft
        #         elif current_gait == "slippery_ground":
        #             active_weights_target = weights_slip
        #         elif current_gait == "muddy_ground":
        #             active_weights_target = weights_muddy
                
        #         print(f"⏱️ [{sim_time:.2f}s] Transitioning to: {current_gait.upper()} (Phase {gait_phase + 1}/{len(gait_sequence)})")
                
        #         # 🚨 TRIGGER CAUTIOUS STEPPING
        #         current_phi = slow_phi_target
        #         print(f"   [Reflex Triggered: Speed dropped to {current_phi:.3f} for safe transition]")
                
        #         gait_phase += 1
        #     else:
        #         print(f"✅ [{sim_time:.2f}s] All combinations completed! Maintaining final gait.")
        
    
    
        joint_targets = {}
        cpg_outputs = {}
        feedback_t = {}
        
        # Update RBFs for ALL 16 Joints smoothly
        for side in LEG_SIDE:
            for index in LEG_INDEX:
                for joint in JOINT_NAMES:  
                    weight_key = f"{index}{side}{joint}"
                    
                    target_weight = active_weights_target[weight_key]
                    prev_weight = rbf_weight_prev[weight_key]
                    
                    # --------------------------------------------------
                    # SMOOTH MASKING KERNEL TRANSITION
                    # --------------------------------------------------
                    target_angle, current_weight = rbf.regenerate_smooth_gated_traj(
                        cpg_output[f'{index}{side}']['cpg_output_0'], 
                        cpg_output[f'{index}{side}']['cpg_output_1'],
                        prev_weight,
                        target_weight,
                        base_alpha=TRANSTION_RATE  # Tweak transition speed here
                    )
                    
                    # Save the new smoothed weight for the next timestep
                    rbf_weight_prev[weight_key] = current_weight
                    
                    actuator_name = f"{index}{side}_J{joint+1}"  
                    joint_targets[actuator_name] = target_angle
                    cpg_outputs[actuator_name] = cpg_output[f'{index}{side}']['cpg_output_0']
            
        phi_list = [data['phi'] for data in cpg_mod_cmd.values()]
        # Pass the flat dictionary to the environment
        feedback_t = env.step(joint_targets, cpg_outputs, all_probabilities, phi_list)
        
        
        
        # ==================================================
        #  ONLINE GAIT SEGMENTATION & CLASSIFICATION
        # ==================================================
        current_cpg_ref = feedback_t['cpg_output'][0]
        
        cycle_completed_this_step = False
        normalized_cycle_dict = {}

        for sensor_key in sensors_to_segment:
            raw_cycle = segmenters[sensor_key].add_data_point(
                current_cpg_ref, 
                feedback_t[sensor_key]  # <-- This passes the full array (length 4 or 16) at once
            )
            
            if raw_cycle is not None:
                cycle_completed_this_step = True
                normalized_cycle_dict[sensor_key] = OnlineGaitSegmenter.normalize_cycle(raw_cycle, num_points=100)
                
        # --------------------------------------------------
        # PUSH TO CLASSIFIER THREAD
        # --------------------------------------------------
        if ENABLE_ESN:
            if cycle_completed_this_step and len(normalized_cycle_dict) == len(sensors_to_segment):
                try:
                    # Put a copy of the dict in the queue so the segmenter can keep working safely
                    cycle_queue.put_nowait((normalized_cycle_dict.copy(), sim_time))
                except queue.Full:
                    print("⚠️ Classifier thread is busy! Dropping frame to maintain real-time performance.")

            
        # ==================================================
        # 🚨 CAUTIOUS STEPPING RECOVERY
        # ==================================================
        # If we are currently slowed down, gradually increase back to base speed
        if current_phi < CPG_PHI:
            current_phi = min(CPG_PHI, current_phi + phi_recovery_rate)
            
            # Apply the recovering speed to the command dictionaries
            for side in LEG_SIDE:
                for index in LEG_INDEX:
                    cpg_mod_cmd[f'{index}{side}']['phi'] = current_phi

        # Update CPGs (This is your existing code)
        for side in LEG_SIDE:
            for index in LEG_INDEX:
                cpg_output[f'{index}{side}'] = cpg_modulated[f'{index}{side}'].modulate_cpg(
                    cpg_mod_cmd[f'{index}{side}']['phi'], 
                    cpg_mod_cmd[f'{index}{side}']['pause_input'], 
                    cpg_mod_cmd[f'{index}{side}']['rewind_input']
                )

        # --------------------------------------------------
        # THREAD-SAFE STATE RETRIEVAL
        # --------------------------------------------------
        with state_lock:
            # ONLY let the classifier override the target if the ESN is actually running!
            if ENABLE_ESN:
                active_weights_target = shared_state['active_weights_target']
                
            all_probabilities = shared_state['all_probabilities']
            
            # Check if the classifier triggered a cautious stepping reflex
            if shared_state['trigger_slowdown']:
                current_phi = slow_phi_target
                print(f"⚠️ Possible terrain change detected! Engaging cautious stepping (Speed dropped).")
                # Reset the flag so it only triggers once
                shared_state['trigger_slowdown'] = False

        # Maintain real-time loop execution
        while (time.perf_counter() - step_start) < dt:
            time.sleep(0.001) 
            
            

    if ENABLE_ESN:
        cycle_queue.put(None)
        classifier_thread.join(timeout=1.0)
    print("Replay finished.")
    env.close()
    
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
        
        if not METRIC_FILE.endswith('.csv'):
            METRIC_FILE += '.csv'
            
        os.makedirs(os.path.dirname(METRIC_FILE), exist_ok=True)
        
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        file_exists = os.path.isfile(METRIC_FILE)
        
        with open(METRIC_FILE, mode="a", newline="") as f:
            fieldnames = ["Timestamp", "Total_Steps", "Imitation_Status"] + list(metric.keys())
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            
            if not file_exists:
                writer.writeheader()
            
            row_data = {
                "Timestamp": timestamp, 
                "Total_Steps": env.step_count,
                "Imitation_Status": "Adaptive_Transition"
            }
            
            for k, v in metric.items():
                row_data[k] = f"{v:.6f}"
                
            writer.writerow(row_data)
            
        print(f"Metrics successfully saved to: {METRIC_FILE}")
