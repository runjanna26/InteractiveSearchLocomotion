import numpy as np
import matplotlib.pyplot as plt


class MiniGNG:
    def __init__(self, n_inputs, max_nodes=50, eps_b=0.2, eps_n=0.006, 
                 max_age=50, lambda_iter=100, alpha=0.5, d=0.995):
        """
        Pure NumPy Growing Neural Gas.
        Optimized for embedded systems via pre-allocated matrices.
        """
        self.n_inputs = n_inputs
        self.max_nodes = max_nodes
        
        # Hyperparameters
        self.eps_b = eps_b          # Learning rate for Best Matching Unit (BMU)
        self.eps_n = eps_n          # Learning rate for neighbors
        self.max_age = max_age      # When to delete an edge
        self.lambda_iter = lambda_iter # Steps between adding a new node
        self.alpha = alpha          # Error decrease after node insertion
        self.d = d                  # Global error decay per step
        
        # Pre-allocated Memory Matrices (Extremely fast on Raspberry Pi)
        self.weights = np.zeros((max_nodes, n_inputs))
        self.errors = np.zeros(max_nodes)
        self.active = np.zeros(max_nodes, dtype=bool)
        
        # Adjacency matrix for edges: -1 means no edge. >= 0 is the age.
        self.edges = np.full((max_nodes, max_nodes), -1.0) 
        
        # Initialize with 2 random nodes
        self.weights[0] = np.random.uniform(-0.1, 0.1, n_inputs)
        self.weights[1] = np.random.uniform(-0.1, 0.1, n_inputs)
        self.active[0] = True
        self.active[1] = True
        self.edges[0, 1] = 0
        self.edges[1, 0] = 0
        
        self.current_nodes = 2
        self.step_count = 0

    def _get_active_indices(self):
        return np.where(self.active)[0]

    def _get_neighbors(self, node_idx):
        return np.where(self.edges[node_idx] != -1)[0]

    def train(self, X, epochs=10):
        """Trains the GNG on a dataset of ESN vectors."""
        for epoch in range(epochs):
            # Shuffle data for better topological mapping
            np.random.shuffle(X)
            
            for x in X:
                self._update(x)

    def _update(self, x):
        """The core Fritzke GNG algorithm step."""
        active_idx = self._get_active_indices()
        
        # 1. Find the 2 nearest nodes (BMU 1 and BMU 2)
        diffs = self.weights[active_idx] - x
        distances = np.linalg.norm(diffs, axis=1)
        
        nearest_relative_idx = np.argsort(distances)[:2]
        s1 = active_idx[nearest_relative_idx[0]]
        s2 = active_idx[nearest_relative_idx[1]]
        
        # 2. Increment age of all edges connected to the BMU (s1)
        neighbors_s1 = self._get_neighbors(s1)
        for n in neighbors_s1:
            self.edges[s1, n] += 1
            self.edges[n, s1] += 1
            
        # 3. Add squared distance to the BMU's error counter
        self.errors[s1] += distances[nearest_relative_idx[0]] ** 2
        
        # 4. Move BMU and its topological neighbors closer to the signal
        self.weights[s1] += self.eps_b * (x - self.weights[s1])
        for n in neighbors_s1:
            self.weights[n] += self.eps_n * (x - self.weights[n])
            
        # 5. Create or reset the edge between s1 and s2
        self.edges[s1, s2] = 0
        self.edges[s2, s1] = 0
        
        # 6. Remove edges that are too old
        for n in neighbors_s1:
            if self.edges[s1, n] > self.max_age:
                self.edges[s1, n] = -1
                self.edges[n, s1] = -1
                
        # 7. Remove isolated nodes (nodes with no edges)
        for idx in active_idx:
            if len(self._get_neighbors(idx)) == 0:
                self.active[idx] = False
                self.current_nodes -= 1
                
        # 8. Insert a new node every lambda iterations
        self.step_count += 1
        if self.step_count % self.lambda_iter == 0 and self.current_nodes < self.max_nodes:
            self._insert_node()
            
        # 9. Decay all errors globally
        self.errors[self.active] *= self.d

    def _insert_node(self):
        """Finds the area with the most error and grows a new node there."""
        active_idx = self._get_active_indices()
        
        # Find node with maximum error (q)
        q = active_idx[np.argmax(self.errors[active_idx])]
        
        # Find the neighbor of q with the maximum error (f)
        neighbors_q = self._get_neighbors(q)
        if len(neighbors_q) == 0:
            return # Should not happen, but safe check
            
        f = neighbors_q[np.argmax(self.errors[neighbors_q])]
        
        # Find an empty slot in our pre-allocated arrays
        inactive_idx = np.where(~self.active)[0]
        if len(inactive_idx) == 0:
            return 
            
        r = inactive_idx[0] # The new node index
        
        # Place new node exactly halfway between q and f
        self.weights[r] = 0.5 * (self.weights[q] + self.weights[f])
        self.active[r] = True
        self.current_nodes += 1
        
        # Update edges: connect r to q and f, and destroy edge between q and f
        self.edges[q, f] = self.edges[f, q] = -1
        self.edges[q, r] = self.edges[r, q] = 0
        self.edges[f, r] = self.edges[r, f] = 0
        
        # Decrease error of q and f, and assign to r
        self.errors[q] *= self.alpha
        self.errors[f] *= self.alpha
        self.errors[r] = self.errors[q]

    @property
    def get_nodes(self):
        """Returns only the valid, active terrain vectors."""
        return self.weights[self.active]

class UnsupervisedESN:
    def __init__(self, n_reservoir=100, leak_rate=0.3, spectral_radius=0.9, random_state=42, max_nodes=50):
        self.n_res = n_reservoir
        self.leak_rate = leak_rate
        self.spectral_radius = spectral_radius
        self.random_state = random_state
        
        self.W_in = None
        self.W_res = None
        
        # Initialize the pure-NumPy MiniGNG
        self.max_nodes = max_nodes
        self.epochs = 10
        self.gng_map = MiniGNG(n_inputs=self.n_res, max_nodes=self.max_nodes)
        self.is_trained = False

    def _initialize_esn(self, n_inputs):
        """Generates the fixed W_in and W_res matrices."""
        np.random.seed(self.random_state)
        self.W_in = np.random.uniform(-0.1, 0.1, (self.n_res, n_inputs))
        
        W_res_raw = np.random.uniform(-1.0, 1.0, (self.n_res, self.n_res))
        eigenvalues = np.linalg.eigvals(W_res_raw)
        self.W_res = W_res_raw * (self.spectral_radius / np.max(np.abs(eigenvalues)))

    def _get_reservoir_photograph(self, cycle):
        """Temporal Compressor: Compresses 100 timesteps into a 100D vector."""
        x = np.zeros(self.n_res)
        n_timesteps = cycle.shape[0]
        
        for t in range(n_timesteps):
            u_t = cycle[t, :]
            x_new = np.tanh(self.W_in @ u_t + self.W_res @ x)
            x = (1 - self.leak_rate) * x + self.leak_rate * x_new
            
        return x 

    def fit(self, X, epochs=10):
        """Unsupervised Training Pipeline using MiniGNG."""
        self.epochs = epochs
        n_samples = len(X)
        n_inputs = X[0].shape[1]
        
        if self.W_in is None:
            self._initialize_esn(n_inputs)

        print(f"1. ESN compressing {n_samples} raw gait cycles...")
        X_compressed = np.zeros((n_samples, self.n_res))
        for i, cycle in enumerate(X):
            X_compressed[i, :] = self._get_reservoir_photograph(cycle)
            
        print(f"2. MiniGNG mapping the topological space...")
        self.gng_map.train(X_compressed, epochs=epochs) 
        self.is_trained = True
        
        # FIXED: Using .current_nodes instead of .nodes!
        n_discovered_terrains = self.gng_map.current_nodes
        print(f"Training Complete! The MiniGNG dynamically grew {n_discovered_terrains} distinct terrain nodes.")

    def predict(self, x_raw):
        """Live Inference: Finds the closest node in the MiniGNG map."""
        if not self.is_trained:
            raise ValueError("The MiniGNG map must be grown before predicting.")
            
        v = self._get_reservoir_photograph(x_raw)
        predicted_node_idx = self._find_bmu(v) 
        return predicted_node_idx
    
    def _find_bmu(self, vector):
        """Ultra-fast vectorized distance calculation."""
        if hasattr(self, 'gng_weights') and self.gng_weights is not None:
            weights_matrix = self.gng_weights
        else:
            # FIXED: Using the @property we wrote in MiniGNG
            weights_matrix = self.gng_map.get_nodes
            
        distances = np.linalg.norm(weights_matrix - vector, axis=1)
        return int(np.argmin(distances))

    def save_model(self, filepath="unsupervised_brain.npz"):
        """Saves the ESN matrices and the MiniGNG node weights."""
        if not self.is_trained:
            raise ValueError("No model parameters to save. Fit the model first.")
		
        gng_weights = self.gng_map.get_nodes
        # self.max_nodes
        # self.epochs
        # self.spectral_radius
            
        np.savez(filepath, 
                 W_in=self.W_in, 
                 W_res=self.W_res, 
                 gng_weights=gng_weights,
                 hyperparams=np.array([
                     self.n_res, 
                     self.leak_rate, 
                     self.max_nodes, 
                     self.epochs,
                     self.spectral_radius
                 ]))
        
        print(f"Hierarchical brain saved to {filepath}")

    def load_model(self, filepath="unsupervised_brain.npz"):
        """Loads matrices and weights, bypassing the need to retrain."""
        data = np.load(filepath)
        self.W_in = data['W_in']
        self.W_res = data['W_res']
        
        self.gng_weights = data['gng_weights']
        
        h_params = data['hyperparams']
        self.n_res = int(h_params[0])
        self.leak_rate = float(h_params[1])
        self.max_nodes = h_params[2]
        self.epochs = h_params[3]
        self.spectral_radius = h_params[4]
        self.is_trained = True
        
        n_discovered_terrains = self.gng_weights.shape[0]
        print(f"\n[SUCCESS] Unsupervised Brain loaded from: '{filepath}'")
        print("-" * 40)
        print("💡 Hyperparameters:")
        print(f"   • Reservoir Size: {self.n_res} neurons")
        print(f"   • Leak Rate:      {self.leak_rate}")
        print(f"   • Discovered Terrains (GNG Nodes): {n_discovered_terrains}")
        print(f"   • Maximum Nodes (GNG Nodes): {self.max_nodes}")
        print(f"   • Number of Epoch (GNG): {self.epochs}")
        print("\n⚙️ Matrix Shapes:")
        print(f"   • W_in  (Input -> Reservoir): {self.W_in.shape}")
        print(f"   • W_res (Reservoir Dynamics): {self.W_res.shape}")
        print(f"   • GNG Map (Nodes x Features): {self.gng_weights.shape}")
        print("-" * 40 + "\n")


