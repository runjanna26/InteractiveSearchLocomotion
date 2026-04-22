import numpy as np
import matplotlib.pyplot as plt

class SupervisedESN:
    def __init__(self, n_reservoir=100, leak_rate=0.3, spectral_radius=0.9, lambda_reg=0.1, random_state=42):
        self.n_res = n_reservoir
        self.leak_rate = leak_rate
        self.spectral_radius = spectral_radius
        self.lambda_reg = lambda_reg
        self.random_state = random_state
        
        # Core matrices
        self.W_in = None
        self.W_res = None
        self.W_out = None
        self.n_classes = None

    def _initialize_matrices(self, n_inputs):
        """Generates the fixed, random W_in and W_res matrices based on input dimensions."""
        np.random.seed(self.random_state)
        
        # Input weights
        self.W_in = np.random.uniform(-0.1, 0.1, (self.n_res, n_inputs))
        
        # Internal reservoir weights scaled by spectral radius
        W_res_raw = np.random.uniform(-1.0, 1.0, (self.n_res, self.n_res))
        eigenvalues = np.linalg.eigvals(W_res_raw)
        max_eigenvalue = np.max(np.abs(eigenvalues))
        self.W_res = W_res_raw * (self.spectral_radius / max_eigenvalue)

    def _get_reservoir_state(self, cycle):
        """Passes a single time-series sequence through the reservoir and returns the final state."""
        x = np.zeros(self.n_res)
        n_timesteps = cycle.shape[0]
        
        for t in range(n_timesteps):
            u_t = cycle[t, :]
            # The Echo State Update Equation
            x_new = np.tanh(self.W_in @ u_t + self.W_res @ x)
            x = (1 - self.leak_rate) * x + self.leak_rate * x_new
            
        return x

    def fit(self, X, y):
        """
        X: List or array of [N x Features] arrays (e.g. 90 x 48 gait cycles)
        y: List or 1D array of integer class labels (e.g. [0, 1, 0, 2...])
        """
        n_samples = len(X)
        n_inputs = X[0].shape[1]
        
        if self.W_in is None:
            self._initialize_matrices(n_inputs)

        print(f"Processing {n_samples} gait cycles through the reservoir...")
        
        # 1. Collect all final states
        X_states = np.zeros((n_samples, self.n_res))
        for i, cycle in enumerate(X):
            X_states[i, :] = self._get_reservoir_state(cycle)
            
        # 2. One-hot encode the target labels
        self.n_classes = len(np.unique(y))
        Y_onehot = np.zeros((n_samples, self.n_classes))
        for i, label in enumerate(y):
            Y_onehot[i, int(label)] = 1.0
            
        # 3. Ridge Regression (Solving for W_out)
        print("Solving for Readout Weights using Ridge Regression...")
        I = np.eye(self.n_res)
        A = X_states.T @ X_states + self.lambda_reg * I
        B = X_states.T @ Y_onehot
        
        # W_out_T = (X^T * X + lambda*I)^-1 * (X^T * Y)
        W_out_T = np.linalg.solve(A, B)
        self.W_out = W_out_T.T # Transpose back to shape (n_classes, n_res)
        
        print(f"ESN Training complete. Readout weight shape: {self.W_out.shape}")
        return self.W_out

    def predict(self, x_raw):
        """
        Predicts the class of a single sequence.
        Returns: Predicted Class ID, Confidence Score (0.0 to 1.0)
        """
        if self.W_out is None:
            raise ValueError("Model must be fitted or loaded before predicting.")
            
        # 1. Extract the latent state
        v = self._get_reservoir_state(x_raw)
        
        # 2. Calculate raw scores
        raw_scores = self.W_out @ v
        
        # 3. Apply Softmax to get a probabilistic confidence score
        exp_scores = np.exp(raw_scores - np.max(raw_scores)) # Shifted for numerical stability
        probabilities = exp_scores / np.sum(exp_scores)
        
        predicted_class = int(np.argmax(probabilities))
        confidence = probabilities[predicted_class]
        
        return predicted_class, confidence

    def save_model(self, filepath="esn_model_params.npz"):
        """Saves the fixed matrices, readout weights, and hyperparameters."""
        if self.W_out is None:
            raise ValueError("No model parameters to save. Fit the model first.")
            
        np.savez(filepath, 
                 W_in=self.W_in, 
                 W_res=self.W_res, 
                 W_out=self.W_out,
                 hyperparams=np.array([self.n_res, self.leak_rate, self.n_classes]))
        
        print(f"ESN parameters saved to {filepath}")

    def load_model(self, filepath="esn_model_params.npz"):
        """Loads matrices and weights, bypassing the need to retrain."""
        data = np.load(filepath)
        self.W_in = data['W_in']
        self.W_res = data['W_res']
        self.W_out = data['W_out']
        
        # Restore parameters
        h_params = data['hyperparams']
        self.n_res = int(h_params[0])
        self.leak_rate = float(h_params[1])
        self.n_classes = int(h_params[2])
        
        # --- Print the Loaded Configuration ---
        print(f"\n[SUCCESS] ESN Model loaded from: '{filepath}'")
        print("-" * 40)
        print("💡 Hyperparameters:")
        print(f"   • Reservoir Size: {self.n_res} neurons")
        print(f"   • Leak Rate:      {self.leak_rate}")
        print(f"   • Target Classes: {self.n_classes}")
        print("\n⚙️ Matrix Shapes:")
        print(f"   • W_in  (Input -> Reservoir): {self.W_in.shape}")
        print(f"   • W_res (Reservoir Dynamics): {self.W_res.shape}")
        print(f"   • W_out (Reservoir -> Output): {self.W_out.shape}")
        print("-" * 40 + "\n")

    def plot_reservoir_dynamics(self, sample_cycle, title="Reservoir Activation Trace"):
        """Visualizes how the internal neurons react to a specific gait cycle."""
        if self.W_in is None:
            self._initialize_matrices(sample_cycle.shape[1])
            
        n_timesteps = sample_cycle.shape[0]
        state_history = np.zeros((n_timesteps, self.n_res))
        x = np.zeros(self.n_res)
        
        # Run the cycle and record history
        for t in range(n_timesteps):
            x_new = np.tanh(self.W_in @ sample_cycle[t, :] + self.W_res @ x)
            x = (1 - self.leak_rate) * x + self.leak_rate * x_new
            state_history[t, :] = x
            
        fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
        
        # Plot 1: The input features (plotting only first 3 features to avoid clutter)
        for i in range(min(3, sample_cycle.shape[1])):
            axes[0].plot(sample_cycle[:, i], label=f'Feature {i}')
        axes[0].set_title("Input Sequence (First 3 Features)")
        axes[0].set_ylabel("Sensor Value")
        axes[0].grid(True, linestyle='--', alpha=0.6)
        
        # Plot 2: The Reservoir Activations
        # Plotting the first 15 neurons so it isn't an unreadable mess
        for i in range(min(15, self.n_res)):
            axes[1].plot(state_history[:, i])
        axes[1].set_title(f"Reservoir Dynamics (Leak Rate = {self.leak_rate})")
        axes[1].set_xlabel("Timestep (Gait Cycle %)")
        axes[1].set_ylabel("Activation (-1 to 1)")
        axes[1].grid(True, linestyle='--', alpha=0.6)
        
        plt.suptitle(title, fontsize=14)
        plt.tight_layout()
        plt.show()