import json
import time
import numpy as np
import os
import subprocess
import signal
import datetime

# import dash
# from dash import dcc, html
# from dash.dependencies import Input, Output
# import plotly.graph_objects as go
# from plotly.subplots import make_subplots
# import threading

from script import StickInsectEnv
from cpg_rbf.cpg_so2 import CPG_SO2, CPG_LOCO
from cpg_rbf.rbf import RBF
from gait_cycle_cut.gait_cycle_cut import OnlineGaitSegmenter
from classification_model_training.models.model_1_esn_rr.env_pred import ESN_RR_Classification

# ======================================================
# CONFIGURATION
# ======================================================

# 0: 'solid_ground', 
# 1: 'soft_ground', 
# 2: 'slippery_ground', 
# 3: 'rough_ground', 
# 4: 'muddy_ground', 
# 5: 'water_surface'
   
gait = "solid_ground"
environment_setup = "water_surface"

SAVE_ROS_BAG = False
ROS_BAG_FILE = f"classification_model_training/rosbags/{gait}_gait_on_{environment_setup}"

SAVE_METRIC_CSV = False
METRIC_FILE = f"classification_model_training/metric/{gait}_gait_on_{environment_setup}"

ENABLE_RENDERING = True


LEG_SIDE    = ['R', 'L']
LEG_INDEX   = ["F", "B"]
JOINT_NAMES = [0, 1, 2, 3]
NUM_KERNELS = 20 

CPG_PHI = 0.05 
SIM_DURATION = 90.0  # seconds

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
    print(environment_classifier_model.ignore_time_column)
    
    # Terrain Mapping Dictionary for printing the prediction
    ID_TO_TERRAIN = {0: 'solid_ground',
                    1: 'slippery_ground',
                    2: 'muddy_ground',
                    3: 'water_surface'}
    # ==========================================
    # 4. INITIALIZE ENVIRONMENT
    # ==========================================
    env = StickInsectEnv(enable_ros=True, render=ENABLE_RENDERING) 
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

    # # ==========================================
    # # 🚨 LIVE PLOTLY DASHBOARD INITIALIZATION
    # # ==========================================
    # # 1. Global dictionary to share data between the Physics Loop and the Web Server
    # live_plotly_data = {
    #     'foot_force': np.zeros(100),
    #     'joint_angle_fb': np.zeros(100),
    #     'joint_velocity_fb': np.zeros(100),
    #     'prediction': "WAITING FOR CYCLE..."
    # }

    # # 2. Build the Dash App
    # app = dash.Dash(__name__)

    # app.layout = html.Div([
    #     html.H2(id='live-title', style={'textAlign': 'center', 'fontFamily': 'Arial', 'color': 'darkred'}),
    #     dcc.Graph(id='live-graph'),
    #     # This timer triggers the graph to request new data every 250 milliseconds
    #     dcc.Interval(id='graph-update', interval=250, n_intervals=0) 
    # ])

    # # 3. Define how the graph updates
    # @app.callback(
    #     [Output('live-graph', 'figure'), Output('live-title', 'children')],
    #     [Input('graph-update', 'n_intervals')]
    # )
    # def update_graph(n):
    #     fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
    #                         subplot_titles=("Foot Force (Front-Right)", "Joint Angle (Front-Right)", "Joint Velocity (Front-Right)"))

    #     x_data = np.linspace(0, 100, 100)

    #     # Pull the latest data from our global dictionary
    #     fig.add_trace(go.Scatter(x=x_data, y=live_plotly_data['foot_force'], mode='lines', line=dict(color='blue', width=2)), row=1, col=1)
    #     fig.add_trace(go.Scatter(x=x_data, y=live_plotly_data['joint_angle_fb'], mode='lines', line=dict(color='green', width=2)), row=2, col=1)
    #     fig.add_trace(go.Scatter(x=x_data, y=live_plotly_data['joint_velocity_fb'], mode='lines', line=dict(color='red', width=2)), row=3, col=1)

    #     fig.update_layout(height=800, showlegend=False, template="plotly_white")
    #     fig.update_xaxes(title_text="Normalized Gait Cycle (%)", row=3, col=1)

    #     title = f"Live ESN Prediction: {live_plotly_data['prediction']}"
    #     return fig, title

    # def run_dash_server():
    #     # Run the web server (reloader must be False to play nicely with MuJoCo threading)
    #     app.run(debug=False, use_reloader=False, port=8050)

    # # 4. Start the Web Server in a background thread
    # dash_thread = threading.Thread(target=run_dash_server, daemon=True)
    # dash_thread.start()
    # print("\n📈 Live Plotly Dashboard running! Open your browser to: http://127.0.0.1:8050\n")
    # ==========================================

    # Map terrain IDs directly to their loaded weights for dynamic switching
    ID_TO_WEIGHTS = {
        0: weights_solid,
        1: weights_slip,
        2: weights_muddy,
        3: weights_water
    }
    
    # Set the initial default gait (e.g., solid ground) before the loop starts
    active_weights_target = weights_solid
    
    # --- DEBOUNCE / ROBUSTNESS TRACKERS ---
    current_gait_class = 0          # Tracks the gait the robot is physically executing right now
    last_predicted_class = -1       # Tracks the last ESN prediction
    consecutive_pred_count = 0      # Counts how many times in a row it was predicted
    REQUIRED_CONSECUTIVE_PREDS = 2  # Change to 10 if you want it to wait longer
    
    TRANSTION_RATE = 0.005
    # --------------------------------------
    
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
            
        # Pass the flat dictionary to the environment
        feedback_t = env.step(joint_targets, cpg_outputs)
        
        
        # ==================================================
        # 🚨 ONLINE GAIT SEGMENTATION & CLASSIFICATION
        # ==================================================
        # 1. Get the Front-Right Leg's CPG output as our "master clock"
        current_cpg_ref = feedback_t['cpg_output'][0]
        
        cycle_completed_this_step = False
        normalized_cycle_dict = {}

        # 2. Feed the ENTIRE time-step arrays into the segmenters simultaneously
        # (Assuming you initialized `segmenters` as a dict: {sensor: OnlineGaitSegmenter()})
        for sensor_key in sensors_to_segment:
            raw_cycle = segmenters[sensor_key].add_data_point(
                current_cpg_ref, 
                feedback_t[sensor_key]  # <-- This passes the full array (length 4 or 16) at once
            )
            
            # 3. If the cycle just finished right now, normalize it to 100 points
            if raw_cycle is not None:
                cycle_completed_this_step = True
                normalized_cycle_dict[sensor_key] = OnlineGaitSegmenter.normalize_cycle(raw_cycle, num_points=100)
                
        # # 4. Trigger the ESN Prediction!
        # if cycle_completed_this_step and len(normalized_cycle_dict) == len(sensors_to_segment):
        #     print(f"\n[Time: {sim_time:.2f}s] ✅ Full Gait Cycle Segmented!")
            
        #     # Reconstruct the 54-column shape by prepending a dummy 'Time' column
        #     time_col = np.linspace(0, 1, 100).reshape(-1, 1) 
            
        #     esn_input_list = []
        #     for s in sensors_to_segment:
        #         sensor_w_time = np.hstack([time_col, normalized_cycle_dict[s]])
        #         esn_input_list.append(sensor_w_time)
                
        #     # Stack horizontally to create the final 2D matrix (100 x 54)
        #     esn_input = np.concatenate(esn_input_list, axis=1)
            
        #     # Predict the terrain 
        #     predicted_class, confidence = environment_classifier_model.predict(esn_input)
        #     predicted_terrain_name = ID_TO_TERRAIN.get(predicted_class, "Unknown Terrain")
            
        #     print(f"🤖 ESN Predicts: {predicted_terrain_name.upper()} (Class {predicted_class}) with {confidence:.2f}% confidence.")
        # # ==================================================

        # 4. Trigger the ESN Prediction!
        if cycle_completed_this_step and len(normalized_cycle_dict) == len(sensors_to_segment):
            # print(f"\n[Time: {sim_time:.2f}s] ✅ Full Gait Cycle Segmented!")
            
            # Reconstruct the 54-column shape by prepending a dummy 'Time' column
            time_col = np.linspace(0, 1, 100).reshape(-1, 1) 
            
            esn_input_list = []
            for s in sensors_to_segment:
                sensor_w_time = np.hstack([normalized_cycle_dict[s], time_col])
                esn_input_list.append(sensor_w_time)
                
            # Stack horizontally to create the final 2D matrix (100 x 54)
            esn_input = np.concatenate(esn_input_list, axis=1)
            
            # Predict the terrain 
            predicted_class, confidence = environment_classifier_model.predict(esn_input)
            predicted_terrain_name = ID_TO_TERRAIN.get(predicted_class, "Unknown Terrain")
            
            print(f"🤖 ESN Predicts: {predicted_terrain_name.upper()} (Class {predicted_class}) with {confidence:.2f}% confidence.")
            
            # ==================================================
            # 🚨 DYNAMIC GAIT SWITCHING WITH DEBOUNCE
            # ==================================================
            # 1. Update the consecutive prediction counter
            if predicted_class == last_predicted_class:
                consecutive_pred_count += 1
            else:
                consecutive_pred_count = 1
                last_predicted_class = predicted_class
                
            print(f"   [Prediction Counter: {consecutive_pred_count} / {REQUIRED_CONSECUTIVE_PREDS}]")

            # 2. Check if we hit the required consecutive threshold AND confidence
            if consecutive_pred_count >= REQUIRED_CONSECUTIVE_PREDS and confidence >= 0.1:
                
                # 3. Only execute the transition if we aren't ALREADY using this gait
                if predicted_class in ID_TO_WEIGHTS and current_gait_class != predicted_class:
                    active_weights_target = ID_TO_WEIGHTS[predicted_class]
                    current_gait_class = predicted_class
                    print(f"🔄 Consistent predictions reached! Adapting gait target to: {predicted_terrain_name.upper()}...")
            # ==================================================
            # ==================================================
            # 🚨 PUSH DATA TO PLOTLY SERVER
            # ==================================================
            # Extract the first column (Front-Right leg/joint) and push to the dictionary
            # live_plotly_data['foot_force'] = normalized_cycle_dict['cpg_output'][:, 0]
            # live_plotly_data['joint_angle_fb'] = normalized_cycle_dict['joint_angle_fb'][:, 0]
            # live_plotly_data['joint_velocity_fb'] = normalized_cycle_dict['joint_velocity_fb'][:, 0]
            
            # # Push the prediction string so the dashboard title updates
            # live_plotly_data['prediction'] = f"{predicted_terrain_name.upper()} ({confidence:.2f}%)"
            # ==================================================



        # Maintain real-time loop execution
        while (time.perf_counter() - step_start) < dt:
            time.sleep(0.001) 
            
            


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
