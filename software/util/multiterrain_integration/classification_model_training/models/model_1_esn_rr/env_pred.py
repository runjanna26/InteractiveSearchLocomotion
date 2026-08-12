import numpy as np
import torch
import pickle
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# ==========================================
# ⚡ THE SECRET SAUCE: JIT COMPILED C++ LOOP
# ==========================================
@torch.jit.script
def _jit_reservoir_loop(x_tensor: torch.Tensor, W_in: torch.Tensor, W_res: torch.Tensor, leak_rate: float, n_res: int, mean_pooling: bool):
    """PyTorch will compile this Python function into highly optimized C++ code."""
    n_timesteps = x_tensor.shape[0]
    x = torch.zeros(n_res, dtype=torch.float32, device=x_tensor.device)
    
    if mean_pooling:
        # Keep a running sum of the states for mean pooling
        x_sum = torch.zeros(n_res, dtype=torch.float32, device=x_tensor.device)
        for t in range(n_timesteps):
            u_t = x_tensor[t, :]
            x_new = torch.tanh(torch.matmul(W_in, u_t) + torch.matmul(W_res, x))
            x = (1.0 - leak_rate) * x + leak_rate * x_new
            x_sum += x
        return x_sum / float(n_timesteps)
    else:
        # Standard execution: only return the final state
        for t in range(n_timesteps):
            u_t = x_tensor[t, :]
            x_new = torch.tanh(torch.matmul(W_in, u_t) + torch.matmul(W_res, x))
            x = (1.0 - leak_rate) * x + leak_rate * x_new
        return x


# =========================================================================================================
#                                   ESN CLASS (With Scaler and mean)
# =========================================================================================================
class ESN_RR_Classification:
    def __init__(self, n_reservoir=100, leak_rate=0.3, spectral_radius=0.9, lambda_reg=0.1, 
                 ignore_time_column=True, output_steps='mean', random_state=42):
        self.n_res = n_reservoir
        self.leak_rate = leak_rate
        self.spectral_radius = spectral_radius
        self.lambda_reg = lambda_reg
        self.ignore_time_column = ignore_time_column 
        
        if output_steps not in ['last', 'mean']:
            raise ValueError("output_steps must be either 'last' or 'mean'")
        self.output_steps = output_steps
        
        self.random_state = random_state
        self.W_in, self.W_res, self.W_out, self.n_classes = None, None, None, None
        self.scaler = None
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.W_in_pt, self.W_res_pt, self.W_out_pt = None, None, None

    def _initialize_matrices(self, n_inputs):
        np.random.seed(self.random_state)
        self.W_in = np.random.uniform(-0.1, 0.1, (self.n_res, n_inputs))
        W_res_raw = np.random.uniform(-1.0, 1.0, (self.n_res, self.n_res))
        eigenvalues = np.linalg.eigvals(W_res_raw)
        self.W_res = W_res_raw * (self.spectral_radius / np.max(np.abs(eigenvalues)))

    def fit(self, X, y):
        if self.ignore_time_column:
            print("Stripping time column from training data...")
            X = [cycle[:, :-1] for cycle in X]

        print("Fitting StandardScaler to training data...")
        self.scaler = StandardScaler()
        X_stacked = np.vstack(X)
        self.scaler.fit(X_stacked)
        X_scaled = [self.scaler.transform(cycle) for cycle in X]
        
        n_samples = len(X_scaled)
        n_inputs = X_scaled[0].shape[1]
        
        if self.W_in is None:
            self._initialize_matrices(n_inputs)

        self.W_in_pt = torch.tensor(self.W_in, dtype=torch.float32, device=self.device)
        self.W_res_pt = torch.tensor(self.W_res, dtype=torch.float32, device=self.device)

        print(f"Processing {n_samples} scaled gait cycles on {self.device.type.upper()} with '{self.output_steps}' pooling...")
        
        mean_pooling = (self.output_steps == 'mean')
        X_states_pt = torch.zeros((n_samples, self.n_res), dtype=torch.float32, device=self.device)
        
        for i, cycle in enumerate(X_scaled):
            cycle_pt = torch.tensor(cycle, dtype=torch.float32, device=self.device)
            X_states_pt[i, :] = _jit_reservoir_loop(cycle_pt, self.W_in_pt, self.W_res_pt, self.leak_rate, self.n_res, mean_pooling)
            
        self.n_classes = len(np.unique(y))
        Y_onehot = np.zeros((n_samples, self.n_classes))
        for i, label in enumerate(y):
            Y_onehot[i, int(label)] = 1.0
            
        Y_onehot_pt = torch.tensor(Y_onehot, dtype=torch.float32, device=self.device)
            
        print("Solving for Readout Weights...")
        I = torch.eye(self.n_res, device=self.device)
        A = X_states_pt.T @ X_states_pt + self.lambda_reg * I
        B = X_states_pt.T @ Y_onehot_pt
        
        self.W_out_pt = torch.linalg.solve(A, B).T 
        self.W_out = self.W_out_pt.cpu().numpy()
        return self.W_out

    def predict(self, x_raw):
        if self.scaler is None or self.W_out_pt is None:
            raise ValueError("Model must be fitted before predicting.")
            
        # 🚨 FIX 1: Safely align features WITHOUT double dropping
        expected_features = self.scaler.n_features_in_
        if x_raw.shape[1] > expected_features:
            x_raw = x_raw[:, :expected_features]
            
        x_scaled = self.scaler.transform(x_raw)
        x_tensor = torch.tensor(x_scaled, dtype=torch.float32, device=self.device)
        
        mean_pooling = (self.output_steps == 'mean')
        x = _jit_reservoir_loop(x_tensor, self.W_in_pt, self.W_res_pt, self.leak_rate, self.n_res, mean_pooling)
            
        raw_scores = torch.matmul(self.W_out_pt, x)
        probabilities = torch.softmax(raw_scores, dim=0).cpu().numpy()
        
        predicted_class = int(np.argmax(probabilities))
        return predicted_class, float(probabilities[predicted_class]), probabilities

    def save_model(self, filepath="esn_model_params.pt"):
        save_dict = {
            'weights': {
                'W_in': torch.tensor(self.W_in),
                'W_res': torch.tensor(self.W_res),
                'W_out': torch.tensor(self.W_out)
            },
            'hyperparams': {
                'n_res': self.n_res, 
                'leak_rate': self.leak_rate, 
                'n_classes': self.n_classes,
                'ignore_time_column': self.ignore_time_column,
                'output_steps': self.output_steps
            },
            'scaler': pickle.dumps(self.scaler)
        }
        torch.save(save_dict, filepath)
        print(f"ESN and Scaler saved to {filepath}")

    def load_model(self, filepath="esn_model_params.pt"):
        # 🚨 FIX: Removed add_safe_globals and changed weights_only to False 
        # to ensure compatibility with older PyTorch versions.
        checkpoint = torch.load(filepath, map_location='cpu', weights_only=False)
        
        self.n_res = int(checkpoint['hyperparams']['n_res'])
        self.leak_rate = float(checkpoint['hyperparams']['leak_rate'])
        self.n_classes = int(checkpoint['hyperparams']['n_classes'])
        self.ignore_time_column = checkpoint['hyperparams'].get('ignore_time_column', False)
        self.output_steps = checkpoint['hyperparams'].get('output_steps', 'last')
        
        self.W_in = checkpoint['weights']['W_in'].cpu().numpy()
        self.W_res = checkpoint['weights']['W_res'].cpu().numpy()
        self.W_out = checkpoint['weights']['W_out'].cpu().numpy()
        
        self.scaler = pickle.loads(checkpoint['scaler'])
        
        self.W_in_pt = checkpoint['weights']['W_in'].to(dtype=torch.float32, device=self.device)
        self.W_res_pt = checkpoint['weights']['W_res'].to(dtype=torch.float32, device=self.device)
        self.W_out_pt = checkpoint['weights']['W_out'].to(dtype=torch.float32, device=self.device)
        

        print(f"\n[SUCCESS] ESN Model loaded from: '{filepath}'")
        print("-" * 50)
        print("💡 Hyperparameters:")
        print(f"   • Reservoir Size (n_res): {self.n_res} neurons")
        print(f"   • Leak Rate:              {self.leak_rate}")
        print(f"   • Target Classes:         {self.n_classes}")
        print(f"   • Ignore Time Column:     {self.ignore_time_column}")
        print(f"   • Pooling Strategy:       '{self.output_steps}'")
        print("\n⚙️ Matrix Shapes:")
        print(f"   • W_in  (Input -> Res):   {self.W_in.shape}")
        print(f"   • W_res (Res Dynamics):   {self.W_res.shape}")
        print(f"   • W_out (Res -> Output):  {self.W_out.shape}")
        print("\n🛠️ System Config:")
        print(f"   • Scaler Initialized:     {self.scaler is not None}")
        print(f"   • Expected Features:      {self.scaler.n_features_in_ if self.scaler else 'N/A'}")
        print(f"   • Target Device:          {self.device.type.upper()}")
        print("-" * 50 + "\n")
    def get_model_size(self):
        """Returns the total number of parameters stored in the ESN matrices."""
        if not hasattr(self, 'W_out') or self.W_out is None:
            raise ValueError("Model is not initialized. Call fit() or load_model() first.")
            
        # Safely handle both NumPy arrays (.size) and PyTorch tensors (.numel())
        in_size = self.W_in.size if hasattr(self.W_in, 'size') else self.W_in.numel()
        res_size = self.W_res.size if hasattr(self.W_res, 'size') else self.W_res.numel()
        out_size = self.W_out.size if hasattr(self.W_out, 'size') else self.W_out.numel()
        
        total_params = out_size
        # total_params = in_size + res_size + out_size
        
        if total_params >= 1_000_000:
            formatted_size = f"{total_params / 1_000_000:.2f}M"
        elif total_params >= 1_000:
            formatted_size = f"{total_params / 1_000:.2f}K"
        else:
            formatted_size = str(total_params)
            
        print(f"Total ESN Parameters (W_in, W_res, W_out): {total_params} ({formatted_size})")
        return total_params
# =========================================================================================================
#                                   ESN CLASS (No Scaler and No mean)
# =========================================================================================================
class ESN_RR_Classification_No_mean:
    def __init__(self, n_reservoir=100, leak_rate=0.3, spectral_radius=0.9, lambda_reg=0.1, random_state=42):
        self.n_res = n_reservoir
        self.leak_rate = leak_rate
        self.spectral_radius = spectral_radius
        self.lambda_reg = lambda_reg
        self.random_state = random_state
        
        self.W_in, self.W_res, self.W_out, self.n_classes = None, None, None, None
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.W_in_pt, self.W_res_pt, self.W_out_pt = None, None, None

    def _initialize_matrices(self, n_inputs):
        np.random.seed(self.random_state)
        self.W_in = np.random.uniform(-0.1, 0.1, (self.n_res, n_inputs))
        W_res_raw = np.random.uniform(-1.0, 1.0, (self.n_res, self.n_res))
        eigenvalues = np.linalg.eigvals(W_res_raw)
        self.W_res = W_res_raw * (self.spectral_radius / np.max(np.abs(eigenvalues)))

    def fit(self, X, y):
        n_samples = len(X)
        n_inputs = X[0].shape[1]
        
        if self.W_in is None:
            self._initialize_matrices(n_inputs)

        self.W_in_pt = torch.tensor(self.W_in, dtype=torch.float32, device=self.device)
        self.W_res_pt = torch.tensor(self.W_res, dtype=torch.float32, device=self.device)

        print(f"Processing {n_samples} gait cycles through the reservoir on {self.device.type.upper()}...")
        
        X_states_pt = torch.zeros((n_samples, self.n_res), dtype=torch.float32, device=self.device)
        for i, cycle in enumerate(X):
            cycle_pt = torch.tensor(cycle, dtype=torch.float32, device=self.device)
            X_states_pt[i, :] = _jit_reservoir_loop(cycle_pt, self.W_in_pt, self.W_res_pt, self.leak_rate, self.n_res, False)
            
        self.n_classes = len(np.unique(y))
        Y_onehot = np.zeros((n_samples, self.n_classes))
        for i, label in enumerate(y):
            Y_onehot[i, int(label)] = 1.0
        Y_onehot_pt = torch.tensor(Y_onehot, dtype=torch.float32, device=self.device)
            
        print("Solving for Readout Weights using Ridge Regression on CUDA...")
        I = torch.eye(self.n_res, device=self.device)
        A = X_states_pt.T @ X_states_pt + self.lambda_reg * I
        B = X_states_pt.T @ Y_onehot_pt
        
        self.W_out_pt = torch.linalg.solve(A, B).T 
        self.W_out = self.W_out_pt.cpu().numpy()
        return self.W_out

    def predict(self, x_raw):
        if self.W_out_pt is None:
            raise ValueError("Model must be fitted or loaded before predicting.")
            
        # 🚨 FIX 2: Protect against dimension mismatch (e.g. 51 vs 50)
        expected_features = self.W_in_pt.shape[1]
        if x_raw.shape[1] > expected_features:
            x_raw = x_raw[:, :expected_features]
            
        x_tensor = torch.tensor(x_raw, dtype=torch.float32, device=self.device)
        x = _jit_reservoir_loop(x_tensor, self.W_in_pt, self.W_res_pt, self.leak_rate, self.n_res, False)
            
        raw_scores = torch.matmul(self.W_out_pt, x)
        probabilities = torch.softmax(raw_scores, dim=0).cpu().numpy()
        
        predicted_class = int(np.argmax(probabilities))
        return predicted_class, float(probabilities[predicted_class])
    
    def save_model(self, filepath="esn_model_params.pt"):
        save_dict = {
            'weights': {
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
        torch.save(save_dict, filepath)
        print(f"ESN parameters saved to {filepath}")

    def load_model(self, filepath="esn_model_params.pt"):
        # 🚨 FIX 3: Add safe globals and weights_only=True
        torch.serialization.add_safe_globals([StandardScaler])
        checkpoint = torch.load(filepath, map_location='cpu', weights_only=True)
        
        self.n_res = int(checkpoint['hyperparams']['n_res'])
        self.leak_rate = float(checkpoint['hyperparams']['leak_rate'])
        self.n_classes = int(checkpoint['hyperparams']['n_classes'])
        
        # 🚨 FIX 4: Add .cpu() to prevent CUDA to NumPy conversion crashes
        self.W_in = checkpoint['weights']['W_in'].cpu().numpy()
        self.W_res = checkpoint['weights']['W_res'].cpu().numpy()
        self.W_out = checkpoint['weights']['W_out'].cpu().numpy()
        
        self.W_in_pt = checkpoint['weights']['W_in'].to(dtype=torch.float32, device=self.device)
        self.W_res_pt = checkpoint['weights']['W_res'].to(dtype=torch.float32, device=self.device)
        self.W_out_pt = checkpoint['weights']['W_out'].to(dtype=torch.float32, device=self.device)
        
        print(f"\n[SUCCESS] ESN No-Mean Model loaded from: '{filepath}'")

    def get_model_size(self):
        if self.W_in is None or self.W_res is None or self.W_out is None:
            raise ValueError("Model matrices are not fully initialized.")
            
        size_out = self.W_out.size
        print(f"Trainable Parameters (W_out): {size_out}")
        return size_out