import numpy as np
from script import StickInsectEnv
from pibb import PIBB 
from logger import TrainingLogger
from cpg_rbf.cpg_so2 import CPG_SO2
from cpg_rbf.rbf import RBF
from cpg_rbf.cpg_so2 import CPG_LOCO

import multiprocessing
import os
import json

# Force NumPy to use only 1 thread per process to prevent CPU thrashing
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
# ======================================================
# CONFIGURATION
# ======================================================
RENDER = False
ROLLOUTS = 14  # Number of parallel rollouts per iteration
SIMULATION_STEPS = 2500  # E.g., 5 seconds at 0.005s timestep
ITERATIONS = 500
NOISE_VARIANCE_INIT = 0.0010
BASE_PARAM_INIT = 0.000
NUM_KERNELS = 20 
# NUM_PARAMETERS = NUM_KERNELS * 8 # 20x8 = 160 parameters (SYMMETRICAL WEIGHT)
# NUM_PARAMETERS = NUM_KERNELS * 16 # 20x16 = 320 parameters for swimming (FULLY INDEPENDENT WEIGHT)
NUM_PARAMETERS_LOAD     = NUM_KERNELS * 16     # (FULLY INDEPENDENT WEIGHT)
NUM_PARAMETERS_LEARN    = NUM_KERNELS * 8      # (SYMMETRICAL WEIGHT)

CPG_PHI = 0.05


# ======================================================
# Initial Variables
# ======================================================
LEG_SIDE    = ['R', 'L']
LEG_INDEX   = ["F", "B"]
JOINT_NAMES = [0, 1, 2, 3]

# The stable baseline posture for the robot (in radians)
STANDING_POSE = {
    'FR': [ np.pi/6, 0.0,  np.pi/9, -2*np.pi/4],
    'BR': [-np.pi/6, 0.0,  np.pi/3, -2*np.pi/3],
    'FL': [-np.pi/6, 0.0,  np.pi/9, -2*np.pi/4],
    'BL': [ np.pi/6, 0.0,  np.pi/3, -2*np.pi/3]
}
# ======================================================
# WORKER FUNCTION (Runs in a separate process)
# ======================================================
def evaluate_rollout(noisy_parameters, left_offsets, right_priors, simulation_steps):
    """
    This function is executed by a separate CPU core.
    It creates its own headless environment to avoid MuJoCo pickling errors.
    """
    # 1. Spin up a local, headless environment for this specific worker
    env = StickInsectEnv(enable_ros=False, render=RENDER)  # Set render=False for headless operation
    
    # 2. Initialize the neural network with the specific noisy parameters
    cpg = CPG_SO2()
    
    rbf = RBF(nc=NUM_KERNELS)

    cpg_one_cycle = cpg.generate_cpg_one_cycle(CPG_PHI)       # lowest phi value
    out0_cpg_one_cycle = cpg_one_cycle['out0_cpg_one_cycle'][:] # Use full cycle of CPG
    out1_cpg_one_cycle = cpg_one_cycle['out1_cpg_one_cycle'][:] # Use full cycle of CPG
    cpg_cycle_length = len(out0_cpg_one_cycle)
    rbf.construct_kernels_with_cpg_one_cycle(out0_cpg_one_cycle, out1_cpg_one_cycle, cpg_cycle_length) # construct kernels with cpg one cycle

    # Control variables
    cpg_modulated = {}
    cpg_output = {}
    cpg_mod_cmd = {}


    # ===============================================================
    # RESIDUAL SYMMETRY LOGIC (With Joint 1 Frozen!)
    # ===============================================================
    imitated_weights = {}
    
    joint_index = 0
    for index in LEG_INDEX: 
        for joint in JOINT_NAMES:
            start_idx = joint_index * NUM_KERNELS
            end_idx = start_idx + NUM_KERNELS
            
            # 🚨 THE FIX: Freeze Joint 1
            if joint == 1:
                # FREEZE: Ignore PIBB completely. Force it to the original prior knowledge.
                learned_weights = np.array(right_priors[start_idx:end_idx])
            else:
                # LEARN: Use PIBB's newly generated noisy parameters (for joints 0, 2, and 3)
                learned_weights = np.array(noisy_parameters[start_idx:end_idx])
            
            # The exact difference between Left and Right from the prior knowledge
            offset_weights = np.array(left_offsets[start_idx:end_idx])
            
            # 1. Assign to RIGHT side
            right_key = f"{index}R{joint}"
            imitated_weights[right_key] = learned_weights
        
            # 2. Assign to LEFT side (Learned update + Original offset)
            left_key = f"{index}L{joint}"
            imitated_weights[left_key] = learned_weights + offset_weights
            
            joint_index += 1
    # ===============================================================
    

    # # ===============================================================
    # # FULLY INDEPENDENT WEIGHT LOGIC (No Symmetry) (Learn)
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
    #             extracted_weights = noisy_parameters[start_idx:end_idx]
                
    #             # Assign them directly to the unique dictionary key
    #             dict_key = f"{index}{side}{joint}"
    #             imitated_weights[dict_key] = extracted_weights
                
    #             joint_index += 1
    # # ===============================================================

    # 2. Setup initial CPG states
    for side in LEG_SIDE:
        for index in LEG_INDEX:
            cpg_modulated[f'{index}{side}']     = CPG_LOCO()
            cpg_output[f'{index}{side}']        = cpg_modulated[f'{index}{side}'].modulate_cpg(CPG_PHI, 0.0, 1.0)
            cpg_mod_cmd[f'{index}{side}']       = {'phi': CPG_PHI, 'pause_input': 0.0, 'rewind_input': 1.0}

    # ===============================================================
    # CPG PHASE SHIFT LOGIC (THE TROT GAIT)
    # ===============================================================
    # We want Front-Left (FL) and Back-Right (BR) to be 180 degrees 
    # out of phase with Front-Right (FR) and Back-Left (BL).
    half_cycle = cpg_cycle_length // 2
    
    for _ in range(half_cycle):
        # Manually step the FL and BR oscillators forward in time
        # before the MuJoCo simulation even begins.
        cpg_output['FL'] = cpg_modulated['FL'].modulate_cpg(CPG_PHI, 0.0, 1.0)
        cpg_output['BR'] = cpg_modulated['BR'].modulate_cpg(CPG_PHI, 0.0, 1.0)

    # 3. Reset environment
    env.reset()
    
    # ======================== 4. Run the simulation loop ========================
    for step in range(simulation_steps):
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
                    target_angle = network_output
                    # target_angle = baseline_angle + network_output
                    
                    # 3. Store it using the exact string format your MuJoCo XML actuators use
                    actuator_name = f"{index}{side}_J{joint+1}"  
                    env_targets[actuator_name] = target_angle
            
        # Pass the flat dictionary to the environment
        env.step(env_targets)
        

    # 5. Calculate and return ONLY the numerical fitness
    fitness_dict = env.calculate_fitness()

    return fitness_dict

# ======================================================
# MAIN LEARNING LOOP
# ======================================================
if __name__ == "__main__":
    # Force 'spawn' to prevent ROS 2 and MuJoCo C-pointer memory corruption
    multiprocessing.set_start_method('spawn')

    # Initialize the Optimizer
    optimizer = PIBB(_rollouts=ROLLOUTS, _h=10.0, _decay=0.995)
    logger = TrainingLogger(log_dir="data/pibb_logs")


    # Use prior knowlegde
    print("Loading Prior Knowledge: Imitated Beetle Gait...")
    # prior_knowledge = np.load('imitated_diving_beetle_walk_forward_weights.npz')
    prior_knowledge = np.load('imitated_diving_beetle_swim_forward_weights_20_kernels.npz')

    # Initialize lists
    base_parameters = []
    left_offsets = []
    right_priors = [] # 🚨 NEW: Create a list for the original right-side priors
    
    LEG_INDEX   = ["F", "B"]
    JOINT_NAMES = [0, 1, 2, 3]
    
    for index in LEG_INDEX:
        for joint in JOINT_NAMES:
            right_key = f"{index}R{joint}" 
            left_key = f"{index}L{joint}"
            
            right_weights = np.array(prior_knowledge[right_key])
            left_weights = np.array(prior_knowledge[left_key])
            
            # Base parameters for PIBB
            base_parameters.extend(right_weights)
            
            # Save the original right priors to send to the worker
            right_priors.extend(right_weights)
            
            # Calculate the difference
            offset = left_weights - right_weights
            left_offsets.extend(offset)
            
    base_parameters = np.array(base_parameters, dtype=np.float64)
    left_offsets = np.array(left_offsets, dtype=np.float64).tolist()
    right_priors = np.array(right_priors, dtype=np.float64).tolist() 
    print(f"Successfully loaded {len(base_parameters)} base parameters and {len(left_offsets)} left offsets!")

    noise_variance = NOISE_VARIANCE_INIT
    
    print("Starting parallel PIBB optimization...")

    # Initialize Smart Checkpoint trackers
    best_max_fitness = -float('inf')
    best_avg_fitness = -float('inf')
    best_min_fitness = -float('inf')
    
    # Ensure the checkpoint directory exists
    CHECKPOINT_DIR = "data/checkpoints"
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    with multiprocessing.Pool(processes=ROLLOUTS, maxtasksperchild=1) as pool:
        for iteration in range(ITERATIONS):
            noise_arr = []
            fitness_arr = []
            noisy_parameters_list = []

            # 3. Generate noise and prep the arguments for the workers
            for k in range(ROLLOUTS):
                noise = np.random.normal(0, np.sqrt(noise_variance), NUM_PARAMETERS_LEARN).tolist()
                noise_arr.append(noise)
                
                noisy_parameters = [base + n for base, n in zip(base_parameters, noise)]
                
                # 🚨 Add left_offsets to the bundled arguments!
                noisy_parameters_list.append((noisy_parameters, left_offsets, right_priors, SIMULATION_STEPS))
            
            # 4. Dispatch tasks and wait for results (blocking)
            # starmap unpacks the tuples in noisy_parameters_list into the evaluate_rollout arguments
            rollout_results = pool.starmap(evaluate_rollout, noisy_parameters_list)

            # --- NEW: Unpack the dictionaries into separate arrays ---
            fitness_arr      = [res["total_reward"] for res in rollout_results]
            fitness_arr_dist = [res["distance_walked"] for res in rollout_results]
            fitness_arr_stab = [res["instability"] for res in rollout_results]
            fitness_arr_coll = [res["collision"] for res in rollout_results]
            fitness_arr_slip = [res["slippage"] for res in rollout_results]

            # 3. Update policy parameters using PIBB
            base_parameters = optimizer.step(fitness_arr, base_parameters, noise_arr)
            
            
            # ==========================================
            # LOG THE DATA AT THE END OF THE ITERATION
            # ==========================================
            logger.log_iteration(
                iteration=iteration,
                fitness_arr=fitness_arr,
                fitness_arr_dist=fitness_arr_dist,  # Pass new arrays
                fitness_arr_stab=fitness_arr_stab,
                fitness_arr_coll=fitness_arr_coll,
                fitness_arr_slip=fitness_arr_slip,
                best_parameters=base_parameters,
                noise_variance=noise_variance
            )
            # ==========================================
            # SMART CHECKPOINTING
            # ==========================================
            current_max = max(fitness_arr)
            current_avg = float(np.mean(fitness_arr))
            current_min = min(fitness_arr)

            # 1. Best Absolute Rollout (Peak capability)
            if current_max > best_max_fitness:
                best_max_fitness = current_max
                with open(os.path.join(CHECKPOINT_DIR, "best_max.json"), "w") as f:
                    json.dump({"iteration": iteration, "fitness": current_max, "parameters": base_parameters}, f)

            # 2. Best Average Rollout (Most stable/consistent policy)
            if current_avg > best_avg_fitness:
                best_avg_fitness = current_avg
                with open(os.path.join(CHECKPOINT_DIR, "best_avg.json"), "w") as f:
                    json.dump({"iteration": iteration, "fitness": current_avg, "parameters": base_parameters}, f)

            # 3. Best Minimum Rollout (Safest/most robust policy)
            if current_min > best_min_fitness:
                best_min_fitness = current_min
                with open(os.path.join(CHECKPOINT_DIR, "best_min.json"), "w") as f:
                    json.dump({"iteration": iteration, "fitness": current_min, "parameters": base_parameters}, f)

            # 4. Decay exploration noise
            noise_variance *= 0.995 

            print(f"Iteration {iteration:03d} | Max Fitness: {max(fitness_arr):.4f} | Avg Fitness: {np.mean(fitness_arr):.4f} | Min Fitness: {min(fitness_arr):.4f} | Noise Variance: {noise_variance:.6f}")

    print("Learning complete. Best parameters extracted.")