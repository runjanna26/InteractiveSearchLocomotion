import numpy as np
import matplotlib.pyplot as plt

class TimeSeriesKMeans:
    def __init__(self, k=2, max_iter=10, window=5):
        self.k = k
        self.max_iter = max_iter
        self.window = window
        self.centroids = None
        self.X_min = None
        self.range_val = None

    def fit(self, X):
        """X: List of [N x 3] arrays (Resampled to fixed length)"""
        n_samples = len(X)
        X_array = np.array(X)
        self.X_min = X_array.min(axis=(0, 1))
        X_max = X_array.max(axis=(0, 1))

        self.range_val = X_max - self.X_min
        self.range_val[self.range_val == 0] = 1.0 # Prevent division by zero for constant features
        
        # Normalize X to [0, 1] range for each feature
        X_scaled = [(cycle - self.X_min) / self.range_val for cycle in X]
        indices = np.random.choice(n_samples, self.k, replace=False)
        self.centroids = [X_scaled[i].copy() for i in indices]
        
        for it in range(self.max_iter):
            clusters = [[] for _ in range(self.k)]
            for x_s in X_scaled:
                best_dist = np.inf
                closest_idx = 0
                for i, c in enumerate(self.centroids):
                    dist = self.fast_multivariate_dtw(x_s, c, self.window, current_best=best_dist)
                    if dist < best_dist:
                        best_dist = dist
                        closest_idx = i
                clusters[closest_idx].append(x_s)
            
            self.centroids = [np.mean(c, axis=0) if c else self.centroids[i] for i, c in enumerate(clusters)]
            print(f"Iteration {it+1} complete")
        return self.centroids
    
    def save_model(self, filepath="gait_model_params.npz"):
        """Saves only the essential parameters to a compressed numpy file."""
        if self.centroids is None:
            raise ValueError("No model parameters to save. Fit the model first.")
        
        np.savez(filepath, 
                 centroids=np.array(self.centroids), 
                 X_min=self.X_min, 
                 range_val=self.range_val,
                 window=np.array([self.window])) # Store window size for consistency
        print(f"Model parameters saved to {filepath}")
        print(f"Saved {self.k} centroids, X_min: {self.X_min}, range_val: {self.range_val}, window: {self.window}")

    def load_model(self, filepath="gait_model_params.npz"):
        """Loads parameters from a compressed numpy file into the class instance."""
        data = np.load(filepath)
        self.centroids = [c for c in data['centroids']]
        self.X_min = data['X_min']
        self.range_val = data['range_val']
        # Restore window size if it was saved
        if 'window' in data:
            self.window = int(data['window'][0])
        
        self.k = len(self.centroids)
        print(f"Model parameters loaded from {filepath}")
        print(f"Loaded {self.k} centroids, X_min: {self.X_min}, range_val: {self.range_val}, window: {self.window}")

    def predict(self, x_raw):
        """Returns: Cluster ID, Uncertainty Score"""
        if self.centroids is None:
            raise ValueError("Model must be fitted or loaded before predicting.")
            
        # Normalize X to [0, 1] range for each feature
        x_scaled = (x_raw - self.X_min) / self.range_val
        dists = [self.fast_multivariate_dtw(x_scaled, c, self.window) for c in self.centroids]
        
        closest_idx = np.argmin(dists)
        d1, d2 = sorted(dists)[:2]
        uncertainty = 1 - (abs(d1 - d2) / (d1 + d2))
        
        return closest_idx, uncertainty
    
    def resample_cycle(self, cycle, target_len=90):
        cycle = np.array(cycle)
        
        # If it's 1D, reshape it to (N, 1) to avoid the unpack error
        if len(cycle.shape) == 1:
            cycle = cycle.reshape(-1, 1)
            
        n_samples, n_features = cycle.shape
        
        # ... rest of your interpolation logic ...
        old_x = np.linspace(0, 1, n_samples)
        new_x = np.linspace(0, 1, target_len)
        
        resampled = np.zeros((target_len, n_features))
        for i in range(n_features):
            resampled[:, i] = np.interp(new_x, old_x, cycle[:, i])
            
        return resampled
    
    def lb_keogh(self, s1, s2, window):
        """
        Very fast O(N) lower bound for DTW. 
        If lb_keogh > current_best_dist, we skip the full DTW.
        """
        lb_dist = 0
        for i in range(len(s1)):
            # Find the min/max in the window of s2
            lower_bound = np.min(s2[max(0, i-window):min(len(s2), i+window+1)], axis=0)
            upper_bound = np.max(s2[max(0, i-window):min(len(s2), i+window+1)], axis=0)
            
            # Check if s1[i] falls outside the s2 envelope
            diff = np.where(s1[i] > upper_bound, s1[i] - upper_bound, 
                            np.where(s1[i] < lower_bound, lower_bound - s1[i], 0))
            lb_dist += np.sum(diff**2)
        return np.sqrt(lb_dist)

    def fast_multivariate_dtw(self, s1, s2, window, current_best=np.inf):
        # 1. Check Lower Bound first
        if self.lb_keogh(s1, s2, window) >= current_best:
            return np.inf 
            
        # 2. If it passes, run optimized DTW (with early exit)
        n, m = len(s1), len(s2)
        dtw_matrix = np.full((n + 1, m + 1), np.inf)
        dtw_matrix[0, 0] = 0
        
        for i in range(1, n + 1):
            lower, upper = max(1, i - window), min(m, i + window)
            row_min = np.inf
            for j in range(lower, upper + 1):
                cost = np.sum((s1[i-1] - s2[j-1])**2)
                dtw_matrix[i, j] = cost + min(dtw_matrix[i-1, j],
                                            dtw_matrix[i, j-1],
                                            dtw_matrix[i-1, j-1])
                row_min = min(row_min, dtw_matrix[i, j])
            
            # Early Exit: If the best path in this row is already worse than best match
            if np.sqrt(row_min) >= current_best:
                return np.inf
                
        return np.sqrt(dtw_matrix[n, m])
    
    def plot_centroids(self, feature_names=['Stiffness', 'Damping', 'Torque FF']):
        # Rescale centroids to original physical units for clarity
        rescaled_centroids = [c * self.range_val + self.X_min for c in self.centroids]
        
        # Create a 3-row subplot (one for each feature)
        fig, axes = plt.subplots(3, 1, figsize=(5, 8), sharex=True)
        labels = [f'Cluster {i}' for i in range(self.k)]

        # Time axis (0 to 100% of the gait cycle)
        t = np.linspace(0, 100, rescaled_centroids[0].shape[0])

        for i, centroid in enumerate(rescaled_centroids):
            for feat_idx in range(3):
                axes[feat_idx].plot(t, centroid[:, feat_idx], 
                                    linewidth=3, 
                                    label=labels[i])
                
                axes[feat_idx].set_ylabel(feature_names[feat_idx], fontsize=12)
                axes[feat_idx].grid(True, linestyle='--', alpha=0.6)

        axes[0].set_title("Self-Supervised Gait Centroids", fontsize=14)
        axes[0].legend(loc='upper right')
        axes[2].set_xlabel("Gait Cycle (%)", fontsize=12)
        
        plt.tight_layout()
        plt.show()