# PIBB find optimal pattern forr each  terrain

## Prior Knowledge
### Configuration
- No standing pose + phi 0.05
ROLLOUTS = 8  # Number of parallel rollouts per iteration
SIMULATION_STEPS = 1500  # E.g., 5 seconds at 0.005s timestep
ITERATIONS = 350
NOISE_VARIANCE_INIT = 0.0005
NUM_KERNELS = 50 
NUM_PARAMETERS = NUM_KERNELS * 16 # Example: 50 RBF neurons * 16 joints = 800 parameters



LOG_FILE = "data/pibb_logs/pibb_training_20260708_033431.json" # <-- Paste your actual log filename here
TARGET_ITERATION = 28    # Set to -1 for the last iteration, or a specific number (e.g., 150)

## Without Priorknowlegde
- 50 Kernels + Standing pose + phi 0.05
ROLLOUTS = 10  # Number of parallel rollouts per iteration
SIMULATION_STEPS = 1500  # E.g., 5 seconds at 0.005s timestep
ITERATIONS = 350
NOISE_VARIANCE_INIT = 0.005
BASE_PARAM_INIT = 0.000
NUM_KERNELS = 50 
NUM_PARAMETERS = NUM_KERNELS * 16 # Example: 50 RBF neurons * 16 joints = 800 parameters

LOG_FILE = "data/pibb_logs/pibb_training_20260708_042156.json" # <-- Paste your actual log filename here
TARGET_ITERATION = 8    # Set to -1 for the last iteration, or a specific number (e.g., 150)

- 20 Kernels + Standing pose + phi 0.05
ROLLOUTS = 10  # Number of parallel rollouts per iteration
SIMULATION_STEPS = 1500  # E.g., 5 seconds at 0.005s timestep
ITERATIONS = 350
NOISE_VARIANCE_INIT = 0.005
BASE_PARAM_INIT = 0.001
NUM_KERNELS = 20 
NUM_PARAMETERS = NUM_KERNELS * 16 # Example: 50 RBF neurons * 16 joints = 800 parameters

LOG_FILE = "data/pibb_logs/pibb_training_20260708_044111.json" # <-- Paste your actual log filename here
TARGET_ITERATION = 29    # Set to -1 for the last iteration, or a specific number (e.g., 150)


- 20 Kernels + Standing pose (quite learning but still moving around)
ROLLOUTS = 10  # Number of parallel rollouts per iteration
SIMULATION_STEPS = 1500  # E.g., 5 seconds at 0.005s timestep
ITERATIONS = 350
NOISE_VARIANCE_INIT = 0.0005
BASE_PARAM_INIT = 0.000
NUM_KERNELS = 20 
NUM_PARAMETERS = NUM_KERNELS * 16 # Example: 50 RBF neurons * 16 joints = 800 parameters
CPG_PHI = 0.02

LOG_FILE = "data/pibb_logs/pibb_training_20260708_135253.json" # <-- Paste your actual log filename here
TARGET_ITERATION = -1    # Set to -1 for the last iteration, or a specific number (e.g., 150)