import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# ==========================================
# 1. The Core PyTorch Module (Hidden from API)
# ==========================================
class _HAPTRNetwork(nn.Module):
    """The internal PyTorch network defined by the paper."""
    def __init__(self, num_classes=8, seq_len=160, input_dim=6, embed_dim=16, 
                 num_heads=8, num_layers=4, dropout=0.1):
        super(_HAPTRNetwork, self).__init__()
        
        # Linear projection (6D -> 16D)
        self.linear_proj = nn.Linear(input_dim, embed_dim)
        
        # Learnable positional encoding
        self.pos_embed = nn.Parameter(torch.randn(1, seq_len, embed_dim))
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=num_heads, 
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # MLP for final classification
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.linear_proj(x)
        x = x + self.pos_embed
        x = self.transformer_encoder(x)
        x = x.mean(dim=1) # Reduce mean over sequence
        x = self.mlp(x)
        return x

# ==========================================
# 2. Your Wrapper Class Format
# ==========================================
class HAPTR_Classifier:
    def __init__(self, variant="Base", epochs=50, batch_size=32, lr=1e-3, random_state=42):
        self.variant = variant
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.random_state = random_state
        
        # Paper Configurations (Light, Base, Large)
        if self.variant == "Light":
            self.n_layers, self.n_heads = 2, 4
        elif self.variant == "Large":
            self.n_layers, self.n_heads = 8, 8
        else: # Default to Base
            self.n_layers, self.n_heads = 4, 8

        self.model = None
        self.n_classes = None
        self.seq_len = None
        self.n_inputs = None
        
        # Hardware acceleration
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _initialize_model(self):
        """Sets up the PyTorch model and optimizer."""
        torch.manual_seed(self.random_state)
        self.model = _HAPTRNetwork(
            num_classes=self.n_classes,
            seq_len=self.seq_len,
            input_dim=self.n_inputs,
            num_heads=self.n_heads,
            num_layers=self.n_layers
        ).to(self.device)
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def fit(self, X, y):
        """
        X: List or array of [N x Features] arrays (e.g. 160 x 6 F/T signals)
        y: List or 1D array of integer class labels
        """
        n_samples = len(X)
        self.seq_len = X[0].shape[0]
        self.n_inputs = X[0].shape[1]
        self.n_classes = len(np.unique(y))
        
        if self.model is None:
            self._initialize_model()

        print(f"Training HAPTR ({self.variant}) on {n_samples} gait cycles using {self.device}...")
        
        # Convert NumPy lists to PyTorch Tensors
        X_tensor = torch.tensor(np.array(X), dtype=torch.float32)
        y_tensor = torch.tensor(np.array(y), dtype=torch.long)
        
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # PyTorch Training Loop
        self.model.train()
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            correct = 0
            
            for batch_X, batch_y in dataloader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item() * batch_X.size(0)
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == batch_y).sum().item()
                
            if (epoch + 1) % 10 == 0 or epoch == 0:
                acc = (correct / n_samples) * 100
                avg_loss = epoch_loss / n_samples
                print(f"Epoch [{epoch+1}/{self.epochs}] | Loss: {avg_loss:.4f} | Accuracy: {acc:.2f}%")
                
        print("HAPTR Training complete.")

    def predict(self, x_raw):
        """
        Predicts the class of a single sequence.
        Returns: Predicted Class ID, Confidence Score (0.0 to 1.0)
        """
        if self.model is None:
            raise ValueError("Model must be fitted or loaded before predicting.")
            
        self.model.eval()
        
        # Convert single sample (160x6) to batch format (1x160x6)
        x_tensor = torch.tensor(x_raw, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            raw_scores = self.model(x_tensor)
            
            # Apply Softmax to get probabilistic confidence score
            probabilities = torch.softmax(raw_scores, dim=1).squeeze(0).cpu().numpy()
            
        predicted_class = int(np.argmax(probabilities))
        confidence = float(probabilities[predicted_class])
        
        return predicted_class, confidence

    def save_model(self, filepath="haptr_model_params.pt"):
        """Saves the PyTorch state dictionary and model parameters."""
        if self.model is None:
            raise ValueError("No model parameters to save. Fit the model first.")
            
        save_dict = {
            'state_dict': self.model.state_dict(),
            'hyperparams': {
                'variant': self.variant,
                'n_classes': self.n_classes,
                'seq_len': self.seq_len,
                'n_inputs': self.n_inputs
            }
        }
        torch.save(save_dict, filepath)
        print(f"HAPTR parameters saved to {filepath}")

    def load_model(self, filepath="haptr_model_params.pt"):
        """Loads weights and topology from a saved file."""
        data = torch.load(filepath, map_location=self.device)
        
        # Restore parameters
        h_params = data['hyperparams']
        self.variant = h_params['variant']
        self.n_classes = h_params['n_classes']
        self.seq_len = h_params['seq_len']
        self.n_inputs = h_params['n_inputs']
        
        if self.variant == "Light":
            self.n_layers, self.n_heads = 2, 4
        elif self.variant == "Large":
            self.n_layers, self.n_heads = 8, 8
        else: 
            self.n_layers, self.n_heads = 4, 8
            
        self._initialize_model()
        self.model.load_state_dict(data['state_dict'])
        self.model.eval()
        
        print(f"\n[SUCCESS] HAPTR Model loaded from: '{filepath}'")
        print("-" * 40)
        print("💡 Hyperparameters:")
        print(f"   • Variant:        {self.variant} (Layers: {self.n_layers}, Heads: {self.n_heads})")
        print(f"   • Target Classes: {self.n_classes}")
        print(f"   • Sequence Length:{self.seq_len} steps")
        print(f"   • Input Channels: {self.n_inputs} (F/T axes)")
        print("-" * 40 + "\n")

    def plot_ft_signals(self, sample_cycle, title="F/T Sensor Sequence"):
        """Visualizes the input 6-axis data directly."""
        fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
        
        # Plot Forces (Assumes first 3 columns are Fx, Fy, Fz)
        axes[0].plot(sample_cycle[:, 0], label='F_x (Red)', color='red')
        axes[0].plot(sample_cycle[:, 1], label='F_y (Green)', color='green')
        axes[0].plot(sample_cycle[:, 2], label='F_z (Blue)', color='blue')
        axes[0].set_title("Forces (N)")
        axes[0].grid(True, linestyle='--', alpha=0.6)
        axes[0].legend(loc='upper right')
        
        # Plot Torques (Assumes next 3 columns are Tx, Ty, Tz)
        axes[1].plot(sample_cycle[:, 3], label='T_x', color='darkred', linestyle='--')
        axes[1].plot(sample_cycle[:, 4], label='T_y', color='darkgreen', linestyle='--')
        axes[1].plot(sample_cycle[:, 5], label='T_z', color='darkblue', linestyle='--')
        axes[1].set_title("Torques (Nm)")
        axes[1].set_xlabel("Timestep (1/400s)")
        axes[1].grid(True, linestyle='--', alpha=0.6)
        axes[1].legend(loc='upper right')
        
        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()

    def get_model_size(self):
        """Returns the total number of trainable parameters in the model."""
        if self.model is None:
            raise ValueError("Model is not initialized. Call fit() or load_model() first.")
            
        total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        # Optional: Format the output nicely
        if total_params >= 1_000_000:
            formatted_size = f"{total_params / 1_000_000:.2f}M"
        elif total_params >= 1_000:
            formatted_size = f"{total_params / 1_000:.2f}K"
        else:
            formatted_size = str(total_params)
            
        print(f"Total Trainable Parameters: {total_params} ({formatted_size})")
        return total_params