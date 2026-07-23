import json
import os
import glob
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# ==========================================
# CONFIGURATION
# ==========================================
LOG_DIR = "data/pibb_logs/"
# Set to a specific filename if you don't want the latest one
SPECIFIC_LOG_FILE = "data/pibb_logs/pibb_training_20260723_133525.json" 

# How often to refresh the plot (in milliseconds)
REFRESH_INTERVAL_MS = 2000  

def get_latest_log_file(directory):
    """Finds the most recently created JSON log file in the directory."""
    list_of_files = glob.glob(os.path.join(directory, '*.json'))
    if not list_of_files:
        return None
    return max(list_of_files, key=os.path.getctime)

# 1. Determine which file to load globally so the animation loop knows what to read
TARGET_LOG_FILE = SPECIFIC_LOG_FILE if SPECIFIC_LOG_FILE else get_latest_log_file(LOG_DIR)

# 2. Setup the figure and axis outside the animation loop
fig, ax = plt.subplots(figsize=(10, 6))

def update_plot(frame):
    """This function is called repeatedly by FuncAnimation."""
    if not TARGET_LOG_FILE or not os.path.exists(TARGET_LOG_FILE):
        return

    # Attempt to read the file. If the training script is currently writing to it, 
    # it might throw a decode error. We just catch it and wait for the next frame.
    try:
        with open(TARGET_LOG_FILE, 'r') as f:
            log_data = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError, ValueError):
        return

    # Extract the data arrays
    iterations = []
    max_fitness = []
    mean_fitness = []
    
    for entry in log_data:
        iterations.append(entry["iteration"])
        metrics = entry["metrics"]
        max_fitness.append(metrics["max_fitness"])
        mean_fitness.append(metrics["mean_fitness"])
        
    # Clear the previous lines
    ax.clear()
    
    # Plot Max and Mean fitness
    ax.plot(iterations, max_fitness
            , label='Max Fitness (Best Rollout)', color='blue', linewidth=2)
    # ax.plot(iterations, mean_fitness, label='Average Fitness (Batch Mean)', color='orange', linewidth=2, linestyle='--')
    
    # Format the plot
    ax.set_title('PIBB Learning Curve (LIVE)', fontsize=16, fontweight='bold')
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Fitness Score', fontsize=12)
    ax.grid(True, linestyle=':', alpha=0.7)
    ax.legend(fontsize=12, loc='upper left')
    
    # Optional: Fill the area between max and mean to visualize variance
    # if len(iterations) > 0:
    #     ax.fill_between(iterations, mean_fitness, max_fitness, color='blue', alpha=0.1)

def start_live_monitor():
    if not TARGET_LOG_FILE:
        print(f"Error: No JSON log files found.")
        return
        
    print(f"Monitoring live data from: {TARGET_LOG_FILE}")
    print("Close the plot window to stop monitoring.")
    
    # Start the animation loop
    ani = animation.FuncAnimation(
        fig, 
        update_plot, 
        interval=REFRESH_INTERVAL_MS, 
        cache_frame_data=False
    )
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    if not os.path.exists(LOG_DIR) and not SPECIFIC_LOG_FILE:
        print(f"Directory {LOG_DIR} does not exist. Please check your path.")
    else:
        start_live_monitor()