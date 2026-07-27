import json
import os
import numpy as np
from datetime import datetime

class TrainingLogger:
    def __init__(self, log_dir="data/logs"):
        """
        Initializes the logger and creates a unique JSON file for the session.
        Mimics the 'data/' directory structure from the CPG-RBFN repository.
        """
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Create a unique filename based on the timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.filepath = os.path.join(self.log_dir, f"pibb_training_{timestamp}.json")
        
        # Initialize an empty history list
        self.history = []
        print(f"[Logger] Initialized. Data will be saved to: {self.filepath}")

    def log_iteration(self, iteration, fitness_arr, fitness_arr_dist, fitness_arr_stab, fitness_arr_coll, fitness_arr_slip, best_parameters, noise_variance):
        """
        Calculates metrics and appends them to the JSON log file.
        """
        entry = {
            "iteration": iteration,
            "metrics": {
                # Total Reward Spread
                "max_fitness": float(np.max(fitness_arr)),
                "mean_fitness": float(np.mean(fitness_arr)),
                "min_fitness": float(np.min(fitness_arr)),
                
                # Sub-component Averages (For plotting)
                "avg_distance": float(np.mean(fitness_arr_dist)),
                "avg_instability": float(np.mean(fitness_arr_stab)),
                "avg_collision": float(np.mean(fitness_arr_coll)),
                "avg_slippage": float(np.mean(fitness_arr_slip))
            },
            "pibb_state": {
                "noise_variance": float(noise_variance)
            },
            "best_parameters": best_parameters
        }

        self.history.append(entry)

        with open(self.filepath, "w") as f:
            json.dump(self.history, f, indent=4)