import numpy as np
import matplotlib.pyplot as plt
import torch

# ==========================================
# ⚡ THE SECRET SAUCE: JIT COMPILED C++ LOOP
# ==========================================
@torch.jit.script
def _jit_reservoir_loop(x_tensor: torch.Tensor, W_in: torch.Tensor, W_res: torch.Tensor, leak_rate: float, n_res: int):
    """PyTorch will compile this Python function into highly optimized C++ code."""
    n_timesteps = x_tensor.shape[0]
    x = torch.zeros(n_res, dtype=torch.float32, device=x_tensor.device)
    
    for t in range(n_timesteps):
        u_t = x_tensor[t, :]
        x_new = torch.tanh(torch.matmul(W_in, u_t) + torch.matmul(W_res, x))
        x = (1.0 - leak_rate) * x + leak_rate * x_new
        
    return x

# ==========================================
# ESN CLASS
# ==========================================
class ESN_RR_Classification:
    def __init__(self, n_reservoir=100, leak_rate=0.3, spectral_radius=0.9, lambda_reg=0.1, random_state=42):
        self.n_res = n_reservoir
        self.leak_rate = leak_rate
        self.spectral_radius = spectral_radius
        self.lambda_reg = lambda_reg
        self.random_state = random_state
        
        # Core matrices (NumPy)
        self.W_in = None
        self.W_res = None
        self.W_out = None
        self.n_classes = None
        
        # 🚨 THE FIX: Pre-initialize PyTorch attributes here
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.W_in_pt = None
        self.W_res_pt = None
        self.W_out_pt = None

    def _initialize_matrices(self, n_inputs):
        np.random.seed(self.random_state)
        self.W_in = np.random.uniform(-0.1, 0.1, (self.n_res, n_inputs))
        W_res_raw = np.random.uniform(-1.0, 1.0, (self.n_res, self.n_res))
        eigenvalues = np.linalg.eigvals(W_res_raw)
        max_eigenvalue = np.max(np.abs(eigenvalues))
        self.W_res = W_res_raw * (self.spectral_radius / max_eigenvalue)

    def _get_reservoir_state(self, cycle):
        x = np.zeros(self.n_res)
        n_timesteps = cycle.shape[0]
        for t in range(n_timesteps):
            u_t = cycle[t, :]
            x_new = np.tanh(self.W_in @ u_t + self.W_res @ x)
            x = (1 - self.leak_rate) * x + self.leak_rate * x_new
        return x

    def fit(self, X, y):
        n_samples = len(X)
        n_inputs = X[0].shape[1]
        
        if self.W_in is None:
            self._initialize_matrices(n_inputs)

        print(f"Processing {n_samples} gait cycles through the reservoir...")
        
        X_states = np.zeros((n_samples, self.n_res))
        for i, cycle in enumerate(X):
            X_states[i, :] = self._get_reservoir_state(cycle)
            
        self.n_classes = len(np.unique(y))
        Y_onehot = np.zeros((n_samples, self.n_classes))
        for i, label in enumerate(y):
            Y_onehot[i, int(label)] = 1.0
            
        print("Solving for Readout Weights using Ridge Regression...")
        I = np.eye(self.n_res)
        A = X_states.T @ X_states + self.lambda_reg * I
        B = X_states.T @ Y_onehot
        
        W_out_T = np.linalg.solve(A, B)
        self.W_out = W_out_T.T 
        
        # 🚨 THE FIX: Immediately push the newly trained weights to PyTorch Tensors 
        # so predict() can be called immediately after fit() without crashing!
        self.W_in_pt = torch.tensor(self.W_in, dtype=torch.float32, device=self.device)
        self.W_res_pt = torch.tensor(self.W_res, dtype=torch.float32, device=self.device)
        self.W_out_pt = torch.tensor(self.W_out, dtype=torch.float32, device=self.device)
        
        print(f"ESN Training complete. Readout weight shape: {self.W_out.shape}")
        return self.W_out

    def predict(self, x_raw):
        """
        Predicts the class of a single sequence using JIT-compiled C++ acceleration.
        Returns: Predicted Class ID, Confidence Score (0.0 to 1.0)
        """
        if self.W_out_pt is None:
            raise ValueError("Model must be fitted or loaded before predicting.")
            
        # 1. Convert input sequence to a tensor
        x_tensor = torch.tensor(x_raw, dtype=torch.float32, device=self.device)
        
        # 2. Run the ultra-fast compiled C++ loop
        x = _jit_reservoir_loop(x_tensor, self.W_in_pt, self.W_res_pt, self.leak_rate, self.n_res)
            
        # 3. Readout and Softmax
        raw_scores = torch.matmul(self.W_out_pt, x)
        probabilities = torch.softmax(raw_scores, dim=0).cpu().numpy()
        
        predicted_class = int(np.argmax(probabilities))
        confidence = float(probabilities[predicted_class])
        
        return predicted_class, confidence

    def save_model(self, filepath="esn_model_params.pt"):
        """Saves the ESN weights and hyperparameters as a PyTorch .pt file."""
        if self.W_out is None:
            raise ValueError("No model parameters to save. Fit the model first.")
            
        # Bundle everything into a neat PyTorch-style dictionary
        save_dict = {
            'weights': {
                # Convert to PyTorch tensors in case they are currently NumPy arrays
                'W_in': torch.tensor(self.W_in) if not torch.is_tensor(self.W_in) else self.W_in,
                'W_res': torch.tensor(self.W_res) if not torch.is_tensor(self.W_res) else self.W_res,
                'W_out': torch.tensor(self.W_out) if not torch.is_tensor(self.W_out) else self.W_out
            },
            'hyperparams': {
                'n_res': self.n_res,
                'leak_rate': self.leak_rate,
                'n_classes': self.n_classes
            }
        }
        
        # Save the dictionary using PyTorch
        torch.save(save_dict, filepath)
        print(f"ESN parameters saved to {filepath}")

    def load_model(self, filepath="esn_model_params.pt"):
        """Loads the ESN weights and hyperparameters from a PyTorch .pt file."""
        # map_location='cpu' ensures it loads safely even if moving from GPU to a CPU machine
        checkpoint = torch.load(filepath, map_location='cpu', weights_only=False)
        
        # Extract hyperparameters from the dictionary
        self.n_res = int(checkpoint['hyperparams']['n_res'])
        self.leak_rate = float(checkpoint['hyperparams']['leak_rate'])
        self.n_classes = int(checkpoint['hyperparams']['n_classes'])
        
        # Extract base weights and convert to numpy 
        # (Keeps compatibility if other parts of your code still expect self.W_in to be numpy)
        self.W_in = checkpoint['weights']['W_in'].numpy()
        self.W_res = checkpoint['weights']['W_res'].numpy()
        self.W_out = checkpoint['weights']['W_out'].numpy()
        
        # Send the tensors directly to the target device
        self.W_in_pt = checkpoint['weights']['W_in'].to(dtype=torch.float32, device=self.device)
        self.W_res_pt = checkpoint['weights']['W_res'].to(dtype=torch.float32, device=self.device)
        self.W_out_pt = checkpoint['weights']['W_out'].to(dtype=torch.float32, device=self.device)
        
        # Formatted terminal output
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
        print(f"   • Target Device:  {self.device.type.upper()}")
        print("-" * 40 + "\n")

    def get_model_size(self):
        """Calculates and returns ONLY the trainable parameters in the ESN (W_out)."""
        if self.W_in is None or self.W_res is None or self.W_out is None:
            raise ValueError("Model matrices are not fully initialized. Call fit() or load_model() first.")
            
        size_in = self.W_in.size
        size_res = self.W_res.size
        size_out = self.W_out.size
        
        trainable_params = size_out
        
        if trainable_params >= 1_000_000:
            formatted_size = f"{trainable_params / 1_000_000:.2f}M"
        elif trainable_params >= 1_000:
            formatted_size = f"{trainable_params / 1_000:.2f}K"
        else:
            formatted_size = str(trainable_params)
            
        print("\n⚙️ ESN Parameter Breakdown:")
        print(f"   • W_in  {self.W_in.shape}: {size_in} weights (Fixed/Non-trainable)")
        print(f"   • W_res {self.W_res.shape}: {size_res} weights (Fixed/Non-trainable)")
        print(f"   • W_out {self.W_out.shape}: {size_out} weights (Trainable)")
        print("-" * 40)
        print(f"Trainable Parameters: {trainable_params} ({formatted_size})\n")
        
        return trainable_params