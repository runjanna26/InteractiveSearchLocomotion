import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# ==========================================
# 1. The Core PyTorch Module (Hidden from API)
# ==========================================
class _Chomp1d(nn.Module):
    """Removes the extra padding added to maintain sequence length and causality."""
    def __init__(self, chomp_size):
        super(_Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class _TemporalBlock(nn.Module):
    """A single TCN block with dilated causal convolutions and residual connection."""
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(_TemporalBlock, self).__init__()
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = _Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = _Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        
        # If input and output channels differ, use a 1x1 conv to match shapes for the residual addition
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class _TCNNetwork(nn.Module):
    """The full TCN network as described in the paper."""
    def __init__(self, num_classes, input_dim, seq_len, num_levels, hidden_units, mlp_units, kernel_size=3, dropout=0.2):
        super(_TCNNetwork, self).__init__()
        
        # 1. Temporal Convolutional Blocks
        layers = []
        num_channels = [hidden_units] * num_levels
        for i in range(num_levels):
            dilation_size = 2 ** i  # Exponentially increasing dilation
            in_channels = input_dim if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            
            layers.append(_TemporalBlock(in_channels, out_channels, kernel_size, stride=1, 
                                         dilation=dilation_size, padding=(kernel_size-1) * dilation_size, 
                                         dropout=dropout))

        self.network = nn.Sequential(*layers)

        # 2. Flatten & Final Classification MLP
        # The paper flattens the entire sequence before passing to the MLP
        self.flatten_size = hidden_units * seq_len

        self.mlp = nn.Sequential(
            nn.Linear(self.flatten_size, mlp_units),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_units, num_classes)
        )

    def forward(self, x):
        # Input shape: (Batch, Seq_Len, Features) -> (B, 160, 6)
        # PyTorch Conv1d expects (Batch, Channels, Seq_Len)
        x = x.transpose(1, 2) 
        
        # Pass through temporal blocks
        y1 = self.network(x)
        
        # Flatten sequence
        y1 = y1.reshape(y1.size(0), -1)
        
        # Classify
        out = self.mlp(y1)
        return out


# ==========================================
# 2. Your Wrapper Class Format
# ==========================================
class TCN_Classifier:
    def __init__(self, variant="Base", epochs=50, batch_size=32, lr=1e-3, random_state=42):
        self.variant = variant
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.random_state = random_state
        
        # Paper Configurations (Light, Base, Large)
        if self.variant == "Light":
            self.num_levels, self.hidden_units, self.mlp_units = 4, 8, 128
        elif self.variant == "Large":
            self.num_levels, self.hidden_units, self.mlp_units = 16, 25, 256
        else: # Default to Base
            self.num_levels, self.hidden_units, self.mlp_units = 8, 16, 256

        self.model = None
        self.n_classes = None
        self.seq_len = None
        self.n_inputs = None
        
        # Hardware acceleration
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _initialize_model(self):
        """Sets up the PyTorch model and optimizer."""
        torch.manual_seed(self.random_state)
        self.model = _TCNNetwork(
            num_classes=self.n_classes,
            input_dim=self.n_inputs,
            seq_len=self.seq_len,
            num_levels=self.num_levels,
            hidden_units=self.hidden_units,
            mlp_units=self.mlp_units
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

        print(f"Training TCN ({self.variant}) on {n_samples} gait cycles using {self.device}...")
        
        X_tensor = torch.tensor(np.array(X), dtype=torch.float32)
        y_tensor = torch.tensor(np.array(y), dtype=torch.long)
        
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
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
                
        print("TCN Training complete.")

    def predict(self, x_raw):
        """
        Predicts the class of a single sequence.
        Returns: Predicted Class ID, Confidence Score (0.0 to 1.0)
        """
        if self.model is None:
            raise ValueError("Model must be fitted or loaded before predicting.")
            
        self.model.eval()
        x_tensor = torch.tensor(x_raw, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            raw_scores = self.model(x_tensor)
            probabilities = torch.softmax(raw_scores, dim=1).squeeze(0).cpu().numpy()
            
        predicted_class = int(np.argmax(probabilities))
        confidence = float(probabilities[predicted_class])
        
        return predicted_class, confidence

    def save_model(self, filepath="tcn_model_params.pt"):
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
        print(f"TCN parameters saved to {filepath}")

    def load_model(self, filepath="tcn_model_params.pt"):
        """Loads weights and topology from a saved file."""
        data = torch.load(filepath, map_location=self.device)
        
        h_params = data['hyperparams']
        self.variant = h_params['variant']
        self.n_classes = h_params['n_classes']
        self.seq_len = h_params['seq_len']
        self.n_inputs = h_params['n_inputs']
        
        # Reconstruct hyperparams based on variant mapping
        if self.variant == "Light":
            self.num_levels, self.hidden_units, self.mlp_units = 4, 8, 128
        elif self.variant == "Large":
            self.num_levels, self.hidden_units, self.mlp_units = 16, 25, 256
        else: 
            self.num_levels, self.hidden_units, self.mlp_units = 8, 16, 256
            
        self._initialize_model()
        self.model.load_state_dict(data['state_dict'])
        self.model.eval()
        
        print(f"\n[SUCCESS] TCN Model loaded from: '{filepath}'")
        print("-" * 40)
        print("💡 Hyperparameters:")
        print(f"   • Variant:        {self.variant} (Conv Levels: {self.num_levels}, Hidden Units: {self.hidden_units})")
        print(f"   • Target Classes: {self.n_classes}")
        print(f"   • Sequence Length:{self.seq_len} steps")
        print(f"   • Input Channels: {self.n_inputs} (F/T axes)")
        print("-" * 40 + "\n")

    def get_model_size(self):
        """Calculates and returns the total number of trainable parameters in the TCN model."""
        if self.model is None:
            raise ValueError("Model is not initialized. Call fit() or load_model() first.")
            
        # Count all parameters that require gradients (are trainable)
        total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        # Format the output nicely
        if total_params >= 1_000_000:
            formatted_size = f"{total_params / 1_000_000:.2f}M"
        elif total_params >= 1_000:
            formatted_size = f"{total_params / 1_000:.2f}K"
        else:
            formatted_size = str(total_params)
            
        print("\n⚙️ TCN Parameter Breakdown:")
        print(f"   • Variant: {self.variant} (Conv Levels = {self.num_levels}, Hidden Units = {self.hidden_units})")
        print(f"   • Total Trainable Parameters: {total_params} ({formatted_size})\n")
        
        return total_params