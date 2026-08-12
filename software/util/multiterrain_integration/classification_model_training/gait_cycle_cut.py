
import numpy as np
from scipy.interpolate import interp1d

class OnlineGaitSegmenter:
    """
    Segments gait data in real-time, one sample at a time.

    This class buffers incoming CPG and gait data. It detects the
    start of a new cycle (a rising zero-crossing in the CPG) to
    dispatch the previously completed cycle.
    """
    
    def __init__(self):
        """Initializes the online segmenter."""
        self.data_buffer = []
        self.last_cpg_val = 0.0
        # We wait for the first zero-crossing to start collecting.
        # This avoids capturing an incomplete partial cycle at the start.
        self.is_collecting = False

    def add_data_point(self, cpg_val: float, data_val) -> np.ndarray | None:
        """
        Processes a single time-step of CPG and gait data.

        Args:
            cpg_val: The CPG signal value at the current time step.
            data_val: The gait data at the current time step.
                      Can be a single float or a list/tuple/array
                      of features (e.g., [joint1, joint2]).

        Returns:
            A 2D NumPy array (shape: (n_steps, n_features)) containing
            one complete gait cycle *only* on the time step that the
            cycle is completed.
            
            Returns `None` on all other time steps.
        """
        
        completed_cycle = None
        
        # --- 1. Detect Rising Zero-Crossing ---
        # (The moment the signal goes from negative to non-negative)
        is_rising_crossing = (self.last_cpg_val < 0) and (cpg_val >= 0)
        
        self.last_cpg_val = cpg_val

        if is_rising_crossing:
            if self.is_collecting:
                # --- 2. A Full Cycle Just Finished ---
                # The buffer holds the complete previous cycle.
                completed_cycle = np.asarray(self.data_buffer)
                
                # Clear the buffer to start collecting the new cycle
                self.data_buffer = []
            else:
                # --- 3. First Crossing Detected ---
                # This is the "true" start. Begin collecting data now.
                self.is_collecting = True

        # --- 4. Buffer Current Data ---
        # If we've passed the first crossing, add the current data
        # to the buffer (which is either new or growing).
        if self.is_collecting:
            self.data_buffer.append(data_val)

        # --- 5. Return the cycle (if one was completed) ---
        if completed_cycle is not None and completed_cycle.size > 0:
             # Ensure output is 2D (timesteps, features)
            if completed_cycle.ndim == 1:
                return completed_cycle.reshape(-1, 1)
            return completed_cycle
        else:
            return None

    def reset(self):
        """Resets the segmenter's internal state."""
        self.data_buffer = []
        self.last_cpg_val = 0.0
        self.is_collecting = False

    @staticmethod
    def normalize_cycle(cycle_data: np.ndarray, num_points: int = 100) -> np.ndarray:
        """
        (Utility Function) Normalizes a *single* completed cycle.

        You would call this on the array returned by `add_data_point`.
        
        Args:
            cycle_data: A 2D array (n_steps, n_features) of one cycle.
            num_points: The target number of points (e.g., 100).

        Returns:
            A 2D array (num_points, n_features) of the normalized cycle.
        """
        if cycle_data.ndim == 1:
            cycle_data = cycle_data.reshape(-1, 1)
            
        cycle_len, num_features = cycle_data.shape
        normalized_cycle = np.zeros((num_points, num_features))

        x_original = np.linspace(0, 1, cycle_len)
        x_normalized = np.linspace(0, 1, num_points)

        for j in range(num_features):
            f = interp1d(x_original, cycle_data[:, j])
            normalized_cycle[:, j] = f(x_normalized)

        return normalized_cycle