import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
import copy

# ==========================================
# 1. The Core PyTorch Module (Hidden from API)
# ==========================================
class _ResLay1D(nn.Module):
    """
    A 1D Convolutional Residual Block with a main branch, a skip branch,
    and downsampling (stride), matching the paper's architecture.
    """
    def __init__(self, in_channels, mid_channels, out_channels, stride=2, kernel_size=5, dropout_rate=0.3, bn_momentum=0.6):
        super(_ResLay1D, self).__init__()
        padding = kernel_size // 2
        
        # Main Branch: Three convolutions with ELU, Dropout, and Batch Normalization
        self.main_branch = nn.Sequential(
            nn.Conv1d(in_channels, mid_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm1d(mid_channels, momentum=bn_momentum),
            nn.ELU(),
            nn.Dropout(dropout_rate),
            
            nn.Conv1d(mid_channels, mid_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm1d(mid_channels, momentum=bn_momentum),
            nn.ELU(),
            nn.Dropout(dropout_rate),
            
            # The final conv in the main branch handles the stride (downsampling)
            nn.Conv1d(mid_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm1d(out_channels, momentum=bn_momentum)
        )
        
        # Skip Branch: Handles the stride and channel mapping to match the main branch addition
        self.skip_branch = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm1d(out_channels, momentum=bn_momentum)
        )
        
        self.final_elu = nn.ELU()

    def forward(self, x):
        main_out = self.main_branch(x)
        skip_out = self.skip_branch(x)
        return self.final_elu(main_out + skip_out)

class _CNNRNNNetwork(nn.Module):
    def __init__(self, num_classes=8, input_dim=6, num_filters=64, rnn_hidden=256, dropout=0.3):
        super(_CNNRNNNetwork, self).__init__()
        
        # --- DYNAMIC SCALING LOGIC ---
        # If num_filters = 64, this generates: 16 -> 32 -> 64
        # If num_filters = 128, this generates: 32 -> 64 -> 128
        out2 = num_filters
        mid2 = max(num_filters // 2, 1) 
        out1 = max(num_filters // 2, 1) 
        mid1 = max(num_filters // 4, 1) 
        
        # --- 1. CNN Feature Extractor ---
        self.reslay1 = _ResLay1D(in_channels=input_dim, mid_channels=mid1, out_channels=out1, stride=2)
        self.reslay2 = _ResLay1D(in_channels=out1, mid_channels=mid2, out_channels=out2, stride=2)

        # --- 2. RNN Temporal Processing ---
        self.gru = nn.GRU(
            input_size=out2,       # Dynamically matches the output of ResLay 2
            hidden_size=rnn_hidden,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )

        # --- 3. Final Predictive Component ---
        self.classifier = nn.Sequential(
            nn.Linear(rnn_hidden, 256), 
            nn.BatchNorm1d(256, momentum=0.6),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        # Transpose for CNN: (Batch, Seq_Len, Features) -> (Batch, Features, Seq_Len)
        x = x.transpose(1, 2) 

        # CNN Extraction
        x = self.reslay1(x)
        x = self.reslay2(x)

        # Transpose for RNN: (Batch, Features, Seq_Len) -> (Batch, Seq_Len, Features)
        x = x.transpose(1, 2)

        # GRU Temporal Processing
        rnn_out, h_n = self.gru(x)

        # Extract and average the final states from both directions
        final_forward = h_n[-2, :, :]  
        final_backward = h_n[-1, :, :] 
        final_feature = (final_forward + final_backward) / 2.0 

        # Classify
        out = self.classifier(final_feature)
        return out

# ==========================================
# 2. Your Wrapper Class Format
# ==========================================
class CNN_RNN_Classifier:
    def __init__(self, num_filters=64, rnn_hidden=256, min_epochs=1000, patience=100, batch_size=256, lr=5e-4, weight_decay=1e-4, random_state=42):
        self.num_filters = num_filters    # Ensure this is stored as an instance variable
        self.rnn_hidden = rnn_hidden
        self.min_epochs = min_epochs
        self.patience = patience
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.random_state = random_state
        
        self.model = None
        self.scaler = StandardScaler()
        self.n_classes = None
        self.seq_len = None
        self.n_inputs = None
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _initialize_model(self):
        """Sets up the PyTorch model, AdamW optimizer, and LR scheduler."""
        torch.manual_seed(self.random_state)
        
        # Pass the dynamic parameters to the network
        self.model = _CNNRNNNetwork(
            num_classes=self.n_classes,
            input_dim=self.n_inputs,
            num_filters=self.num_filters, 
            rnn_hidden=self.rnn_hidden
        ).to(self.device)
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.995)

    def fit(self, X, y):
        n_samples = len(X)
        self.seq_len = X[0].shape[0]
        self.n_inputs = X[0].shape[1]
        self.n_classes = len(np.unique(y))
        
        # Flatten the 3D array to 2D to fit the scaler across all time steps
        X_np = np.array(X)
        X_2d = X_np.reshape(-1, self.n_inputs)
        X_scaled_2d = self.scaler.fit_transform(X_2d)
        
        # Reshape back to 3D for the network
        X_scaled = X_scaled_2d.reshape(n_samples, self.seq_len, self.n_inputs)
        
        if self.model is None:
            self._initialize_model()

        print(f"Training CNN-RNN on {n_samples} gait cycles using {self.device}...")
        
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        y_tensor = torch.tensor(np.array(y), dtype=torch.long)
        
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        self.model.train()
        
        best_loss = float('inf')
        best_model_weights = None
        epochs_no_improve = 0
        current_epoch = 0
        
        # Loop handles minimum 1000 epochs, plus early stopping patience
        while True:
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
            
            self.scheduler.step()
            
            avg_loss = epoch_loss / n_samples
            acc = (correct / n_samples) * 100
            
            current_epoch += 1
            
            if current_epoch % 50 == 0 or current_epoch == 1:
                print(f"Epoch [{current_epoch}] | Loss: {avg_loss:.4f} | Accuracy: {acc:.2f}% | LR: {self.scheduler.get_last_lr()[0]:.6f}")
            
            # Start tracking for early stopping ONLY after min_epochs
            if current_epoch > self.min_epochs:
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    best_model_weights = copy.deepcopy(self.model.state_dict())
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                    
                if epochs_no_improve >= self.patience:
                    print(f"🛑 Early stopping triggered at epoch {current_epoch}. No progress for {self.patience} epochs.")
                    self.model.load_state_dict(best_model_weights)
                    break
                    
        print("✅ CNN-RNN Training complete.")

    def predict(self, x_raw):
        if self.model is None:
            raise ValueError("Model must be fitted or loaded before predicting.")
            
        self.model.eval() 
        
        x_scaled = self.scaler.transform(np.array(x_raw))
        x_tensor = torch.tensor(x_scaled, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            raw_scores = self.model(x_tensor)
            probabilities = torch.softmax(raw_scores, dim=1).squeeze(0).cpu().numpy()
            
        predicted_class = int(np.argmax(probabilities))
        confidence = float(probabilities[predicted_class])
        
        return predicted_class, confidence

    def save_model(self, filepath="cnn_rnn_model_params.pt"):
        if self.model is None:
            raise ValueError("No model parameters to save. Fit the model first.")
            
        save_dict = {
            'state_dict': self.model.state_dict(),
            'scaler': self.scaler, 
            'hyperparams': {
                'rnn_hidden': self.rnn_hidden,
                'n_classes': self.n_classes,
                'seq_len': self.seq_len,
                'n_inputs': self.n_inputs
            }
        }
        torch.save(save_dict, filepath)
        print(f"CNN-RNN parameters and scaler saved to {filepath}")

    def load_model(self, filepath="cnn_rnn_model_params.pt"):
        data = torch.load(filepath, map_location=self.device, weights_only=False)
        
        h_params = data['hyperparams']
        self.rnn_hidden = h_params['rnn_hidden']
        self.n_classes = h_params['n_classes']
        self.seq_len = h_params['seq_len']
        self.n_inputs = h_params['n_inputs']
        
        self.scaler = data['scaler'] 
        
        self._initialize_model()
        self.model.load_state_dict(data['state_dict'])
        self.model.eval()
        
        print("-" * 40)
        print(f"\n[SUCCESS] CNN-RNN Model loaded from: '{filepath}'")
        print("💡 Hyperparameters:")
        print(f"   • CNN Filters:    {self.num_filters}")
        print(f"   • GRU Hidden Dim: {self.rnn_hidden}")
        print(f"   • Target Classes: {self.n_classes}")
        print(f"   • Sequence Length:{self.seq_len} steps")
        print(f"   • Input Channels: {self.n_inputs} (F/T axes)")
        print("-" * 40 + "\n")

    def get_model_size(self):
        if self.model is None:
            raise ValueError("Model is not initialized. Call fit() or load_model() first.")
            
        total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        if total_params >= 1_000_000:
            formatted_size = f"{total_params / 1_000_000:.2f}M"
        elif total_params >= 1_000:
            formatted_size = f"{total_params / 1_000:.2f}K"
        else:
            formatted_size = str(total_params)
            
        print("\n⚙️ CNN-RNN Parameter Breakdown:")
        print(f"   • Total Trainable Parameters: {total_params} ({formatted_size})\n")
        
        return total_params

# import numpy as np
# import torch
# import torch.nn as nn
# from torch.utils.data import TensorDataset, DataLoader
# from sklearn.preprocessing import StandardScaler

# # ==========================================
# # 1. The Core PyTorch Module (Hidden from API)
# # ==========================================
# class _ResidualBlock1D(nn.Module):
#     """A 1D Convolutional Residual Block with Batch Normalization."""
#     def __init__(self, channels):
#         super(_ResidualBlock1D, self).__init__()
#         self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
#         self.bn1 = nn.BatchNorm1d(channels)
#         self.relu = nn.ReLU()
#         self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
#         self.bn2 = nn.BatchNorm1d(channels)

#     def forward(self, x):
#         residual = x
#         out = self.relu(self.bn1(self.conv1(x)))
#         out = self.bn2(self.conv2(out))
#         out += residual
#         return self.relu(out)

# class _CNNRNNNetwork(nn.Module):
#     """The CNN-RNN network as described in the baseline section."""
#     def __init__(self, num_classes=8, input_dim=6, num_filters=64, rnn_hidden=128, dropout=0.3):
#         super(_CNNRNNNetwork, self).__init__()
        
#         # --- 1. CNN Feature Extractor ---
#         self.initial_conv = nn.Sequential(
#             nn.Conv1d(in_channels=input_dim, out_channels=num_filters, kernel_size=5, padding=2),
#             nn.BatchNorm1d(num_filters),
#             nn.ReLU()
#         )
        
#         self.res_block1 = _ResidualBlock1D(num_filters)
#         self.res_block2 = _ResidualBlock1D(num_filters)

#         # --- 2. RNN Temporal Processing ---
#         self.gru = nn.GRU(
#             input_size=num_filters,
#             hidden_size=rnn_hidden,
#             num_layers=2,
#             batch_first=True,
#             bidirectional=True,
#             dropout=dropout
#         )

#         # --- 3. Final Classification MLP ---
#         self.classifier = nn.Sequential(
#             nn.Dropout(dropout),
#             nn.Linear(rnn_hidden * 2, 64),
#             nn.ReLU(),
#             nn.Linear(64, num_classes)
#         )

#     def forward(self, x):
#         # PyTorch Conv1d expects (Batch, Channels, Length)
#         x = x.transpose(1, 2) 

#         # CNN Extraction
#         x = self.initial_conv(x)
#         x = self.res_block1(x)
#         x = self.res_block2(x)

#         # Prepare for RNN: (Batch, Channels, Length) -> (Batch, Length, Channels)
#         x = x.transpose(1, 2)

#         # GRU Temporal Processing
#         rnn_out, h_n = self.gru(x)

#         # --- FIX: Extract the true final states from both forward and backward directions ---
#         final_forward = h_n[-2, :, :]  # Last layer, forward direction
#         final_backward = h_n[-1, :, :] # Last layer, backward direction
        
#         final_feature = torch.cat([final_forward, final_backward], dim=1)

#         # Classify
#         out = self.classifier(final_feature)
#         return out


# # ==========================================
# # 2. Your Wrapper Class Format
# # ==========================================
# class CNN_RNN_Classifier:
#     def __init__(self, num_filters=64, rnn_hidden=128, epochs=50, batch_size=32, lr=1e-3, random_state=42):
#         self.num_filters = num_filters
#         self.rnn_hidden = rnn_hidden
#         self.epochs = epochs
#         self.batch_size = batch_size
#         self.lr = lr
#         self.random_state = random_state
        
#         self.model = None
#         self.scaler = StandardScaler() # NEW: Initialize the scaler
#         self.n_classes = None
#         self.seq_len = None
#         self.n_inputs = None
        
#         # Hardware acceleration
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     def _initialize_model(self):
#         """Sets up the PyTorch model and optimizer."""
#         torch.manual_seed(self.random_state)
#         self.model = _CNNRNNNetwork(
#             num_classes=self.n_classes,
#             input_dim=self.n_inputs,
#             num_filters=self.num_filters,
#             rnn_hidden=self.rnn_hidden
#         ).to(self.device)
        
#         self.criterion = nn.CrossEntropyLoss()
#         self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

#     def fit(self, X, y):
#         """
#         X: List or array of [N x Features] arrays
#         y: List or 1D array of integer class labels
#         """
#         n_samples = len(X)
#         self.seq_len = X[0].shape[0]
#         self.n_inputs = X[0].shape[1]
#         self.n_classes = len(np.unique(y))
        
#         # --- NEW: APPLY SCALING ---
#         # Flatten the 3D array to 2D to fit the scaler across all time steps
#         X_np = np.array(X)
#         X_2d = X_np.reshape(-1, self.n_inputs)
#         X_scaled_2d = self.scaler.fit_transform(X_2d)
        
#         # Reshape back to 3D for the network
#         X_scaled = X_scaled_2d.reshape(n_samples, self.seq_len, self.n_inputs)
        
#         if self.model is None:
#             self._initialize_model()

#         print(f"Training CNN-RNN on {n_samples} gait cycles using {self.device}...")
        
#         X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
#         y_tensor = torch.tensor(np.array(y), dtype=torch.long)
        
#         dataset = TensorDataset(X_tensor, y_tensor)
#         dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
#         self.model.train()
#         for epoch in range(self.epochs):
#             epoch_loss = 0.0
#             correct = 0
            
#             for batch_X, batch_y in dataloader:
#                 batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
#                 self.optimizer.zero_grad()
#                 outputs = self.model(batch_X)
                
#                 loss = self.criterion(outputs, batch_y)
#                 loss.backward()
#                 self.optimizer.step()
                
#                 epoch_loss += loss.item() * batch_X.size(0)
#                 _, predicted = torch.max(outputs.data, 1)
#                 correct += (predicted == batch_y).sum().item()
                
#             if (epoch + 1) % 10 == 0 or epoch == 0:
#                 acc = (correct / n_samples) * 100
#                 avg_loss = epoch_loss / n_samples
#                 print(f"Epoch [{epoch+1}/{self.epochs}] | Loss: {avg_loss:.4f} | Accuracy: {acc:.2f}%")
                
#         print("CNN-RNN Training complete.")

#     def predict(self, x_raw):
#         """
#         Predicts the class of a single sequence.
#         Returns: Predicted Class ID, Confidence Score (0.0 to 1.0)
#         """
#         if self.model is None:
#             raise ValueError("Model must be fitted or loaded before predicting.")
            
#         self.model.eval() # Ensure dropout and batchnorm are frozen
        
#         # --- NEW: SCALE THE INCOMING PREDICTION DATA ---
#         x_scaled = self.scaler.transform(np.array(x_raw))
#         x_tensor = torch.tensor(x_scaled, dtype=torch.float32).unsqueeze(0).to(self.device)
        
#         with torch.no_grad():
#             raw_scores = self.model(x_tensor)
#             probabilities = torch.softmax(raw_scores, dim=1).squeeze(0).cpu().numpy()
            
#         predicted_class = int(np.argmax(probabilities))
#         confidence = float(probabilities[predicted_class])
        
#         return predicted_class, confidence

#     def save_model(self, filepath="cnn_rnn_model_params.pt"):
#         """Saves the PyTorch state dictionary, scaler, and hyperparameters."""
#         if self.model is None:
#             raise ValueError("No model parameters to save. Fit the model first.")
            
#         save_dict = {
#             'state_dict': self.model.state_dict(),
#             'scaler': self.scaler,  # NEW: Save the fitted scaler
#             'hyperparams': {
#                 'num_filters': self.num_filters,
#                 'rnn_hidden': self.rnn_hidden,
#                 'n_classes': self.n_classes,
#                 'seq_len': self.seq_len,
#                 'n_inputs': self.n_inputs
#             }
#         }
#         torch.save(save_dict, filepath)
#         print(f"CNN-RNN parameters and scaler saved to {filepath}")

#     def load_model(self, filepath="cnn_rnn_model_params.pt"):
#         """Loads weights, scaler, and topology from a saved file."""
#         data = torch.load(filepath, map_location=self.device, weights_only=False)
        
#         h_params = data['hyperparams']
#         self.num_filters = h_params['num_filters']
#         self.rnn_hidden = h_params['rnn_hidden']
#         self.n_classes = h_params['n_classes']
#         self.seq_len = h_params['seq_len']
#         self.n_inputs = h_params['n_inputs']
        
#         self.scaler = data['scaler'] # NEW: Load the fitted scaler
        
#         self._initialize_model()
#         self.model.load_state_dict(data['state_dict'])
#         self.model.eval()
        
#         print(f"\n[SUCCESS] CNN-RNN Model loaded from: '{filepath}'")
#         print("-" * 40)
#         print("💡 Hyperparameters:")
#         print(f"   • CNN Filters:    {self.num_filters}")
#         print(f"   • GRU Hidden Dim: {self.rnn_hidden}")
#         print(f"   • Target Classes: {self.n_classes}")
#         print(f"   • Sequence Length:{self.seq_len} steps")
#         print(f"   • Input Channels: {self.n_inputs} (F/T axes)")
#         print("-" * 40 + "\n")

#     def get_model_size(self):
#         """Calculates and returns the total number of trainable parameters in the CNN-RNN model."""
#         if self.model is None:
#             raise ValueError("Model is not initialized. Call fit() or load_model() first.")
            
#         # Count all parameters that require gradients (are trainable)
#         total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
#         # Format the output nicely
#         if total_params >= 1_000_000:
#             formatted_size = f"{total_params / 1_000_000:.2f}M"
#         elif total_params >= 1_000:
#             formatted_size = f"{total_params / 1_000:.2f}K"
#         else:
#             formatted_size = str(total_params)
            
#         print("\n⚙️ CNN-RNN Parameter Breakdown:")
#         print(f"   • Base Architecture: CNN Filters = {self.num_filters}, GRU Hidden = {self.rnn_hidden}")
#         print(f"   • Total Trainable Parameters: {total_params} ({formatted_size})\n")
        
#         return total_params


# import numpy as np
# import matplotlib.pyplot as plt
# import torch
# import torch.nn as nn
# from torch.utils.data import TensorDataset, DataLoader

# # ==========================================
# # 1. The Core PyTorch Module (Hidden from API)
# # ==========================================
# class _ResidualBlock1D(nn.Module):
#     """A 1D Convolutional Residual Block with Batch Normalization."""
#     def __init__(self, channels):
#         super(_ResidualBlock1D, self).__init__()
#         self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
#         self.bn1 = nn.BatchNorm1d(channels)
#         self.relu = nn.ReLU()
#         self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
#         self.bn2 = nn.BatchNorm1d(channels)

#     def forward(self, x):
#         residual = x
#         out = self.relu(self.bn1(self.conv1(x)))
#         out = self.bn2(self.conv2(out))
#         out += residual
#         return self.relu(out)

# class _CNNRNNNetwork(nn.Module):
#     """The CNN-RNN network as described in the baseline section."""
#     def __init__(self, num_classes=8, input_dim=6, num_filters=64, rnn_hidden=128, dropout=0.3):
#         super(_CNNRNNNetwork, self).__init__()
        
#         # --- 1. CNN Feature Extractor ---
#         # Initial convolution to project the 6-axis data into higher dimensional filters
#         self.initial_conv = nn.Sequential(
#             nn.Conv1d(in_channels=input_dim, out_channels=num_filters, kernel_size=5, padding=2),
#             nn.BatchNorm1d(num_filters),
#             nn.ReLU()
#         )
        
#         # Residual layers as mentioned in the paper
#         self.res_block1 = _ResidualBlock1D(num_filters)
#         self.res_block2 = _ResidualBlock1D(num_filters)

#         # --- 2. RNN Temporal Processing ---
#         # Two bidirectional layers with GRU cells
#         self.gru = nn.GRU(
#             input_size=num_filters,
#             hidden_size=rnn_hidden,
#             num_layers=2,
#             batch_first=True,
#             bidirectional=True,
#             dropout=dropout
#         )

#         # --- 3. Final Classification MLP ---
#         # Bidirectional means output hidden size is doubled (rnn_hidden * 2)
#         self.classifier = nn.Sequential(
#             nn.Dropout(dropout),
#             nn.Linear(rnn_hidden * 2, 64),
#             nn.ReLU(),
#             nn.Linear(64, num_classes)
#         )
#     def forward(self, x):
#         # PyTorch Conv1d expects (Batch, Channels, Length)
#         x = x.transpose(1, 2) 

#         # CNN Extraction
#         x = self.initial_conv(x)
#         x = self.res_block1(x)
#         x = self.res_block2(x)

#         # Prepare for RNN: (Batch, Channels, Length) -> (Batch, Length, Channels)
#         x = x.transpose(1, 2)

#         # GRU Temporal Processing
#         rnn_out, h_n = self.gru(x)

#         # --- FIX: Extract the true final states from both directions ---
#         # h_n shape: (num_layers * num_directions, batch, hidden_size)
#         final_forward = h_n[-2, :, :]  # Last layer, forward direction
#         final_backward = h_n[-1, :, :] # Last layer, backward direction
        
#         final_feature = torch.cat([final_forward, final_backward], dim=1)

#         # Classify
#         out = self.classifier(final_feature)
#         return out
#     # def forward(self, x):
#     #     # Input shape: (Batch, Seq_Len, Features) -> (B, 160, 6)
#     #     # PyTorch Conv1d expects (Batch, Channels, Length)
#     #     x = x.transpose(1, 2) # Now (B, 6, 160)

#     #     # CNN Extraction
#     #     x = self.initial_conv(x)
#     #     x = self.res_block1(x)
#     #     x = self.res_block2(x)

#     #     # Prepare for RNN: (Batch, Channels, Length) -> (Batch, Length, Channels)
#     #     x = x.transpose(1, 2)

#     #     # GRU Temporal Processing
#     #     # rnn_out shape: (Batch, Seq_Len, hidden_dim * 2)
#     #     rnn_out, _ = self.gru(x)

#     #     # Extract the features from the final time step
#     #     final_feature = rnn_out[:, -1, :]

#     #     # Classify
#     #     out = self.classifier(final_feature)
#     #     return out


# # ==========================================
# # 2. Your Wrapper Class Format
# # ==========================================
# class CNN_RNN_Classifier:
#     def __init__(self, num_filters=64, rnn_hidden=128, epochs=50, batch_size=32, lr=1e-3, random_state=42):
#         self.num_filters = num_filters
#         self.rnn_hidden = rnn_hidden
#         self.epochs = epochs
#         self.batch_size = batch_size
#         self.lr = lr
#         self.random_state = random_state
        
#         self.model = None
#         self.n_classes = None
#         self.seq_len = None
#         self.n_inputs = None
        
#         # Hardware acceleration
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     def _initialize_model(self):
#         """Sets up the PyTorch model and optimizer."""
#         torch.manual_seed(self.random_state)
#         self.model = _CNNRNNNetwork(
#             num_classes=self.n_classes,
#             input_dim=self.n_inputs,
#             num_filters=self.num_filters,
#             rnn_hidden=self.rnn_hidden
#         ).to(self.device)
        
#         self.criterion = nn.CrossEntropyLoss()
#         self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

#     def fit(self, X, y):
#         """
#         X: List or array of [N x Features] arrays (e.g. 160 x 6 F/T signals)
#         y: List or 1D array of integer class labels
#         """
#         n_samples = len(X)
#         self.seq_len = X[0].shape[0]
#         self.n_inputs = X[0].shape[1]
#         self.n_classes = len(np.unique(y))
        
#         if self.model is None:
#             self._initialize_model()

#         print(f"Training CNN-RNN on {n_samples} gait cycles using {self.device}...")
        
#         X_tensor = torch.tensor(np.array(X), dtype=torch.float32)
#         y_tensor = torch.tensor(np.array(y), dtype=torch.long)
        
#         dataset = TensorDataset(X_tensor, y_tensor)
#         dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
#         self.model.train()
#         for epoch in range(self.epochs):
#             epoch_loss = 0.0
#             correct = 0
            
#             for batch_X, batch_y in dataloader:
#                 batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
#                 self.optimizer.zero_grad()
#                 outputs = self.model(batch_X)
                
#                 loss = self.criterion(outputs, batch_y)
#                 loss.backward()
#                 self.optimizer.step()
                
#                 epoch_loss += loss.item() * batch_X.size(0)
#                 _, predicted = torch.max(outputs.data, 1)
#                 correct += (predicted == batch_y).sum().item()
                
#             if (epoch + 1) % 10 == 0 or epoch == 0:
#                 acc = (correct / n_samples) * 100
#                 avg_loss = epoch_loss / n_samples
#                 print(f"Epoch [{epoch+1}/{self.epochs}] | Loss: {avg_loss:.4f} | Accuracy: {acc:.2f}%")
                
#         print("CNN-RNN Training complete.")

#     def predict(self, x_raw):
#         """
#         Predicts the class of a single sequence.
#         Returns: Predicted Class ID, Confidence Score (0.0 to 1.0)
#         """
#         if self.model is None:
#             raise ValueError("Model must be fitted or loaded before predicting.")
            
#         self.model.eval()
#         x_tensor = torch.tensor(x_raw, dtype=torch.float32).unsqueeze(0).to(self.device)
        
#         with torch.no_grad():
#             raw_scores = self.model(x_tensor)
#             probabilities = torch.softmax(raw_scores, dim=1).squeeze(0).cpu().numpy()
            
#         predicted_class = int(np.argmax(probabilities))
#         confidence = float(probabilities[predicted_class])
        
#         return predicted_class, confidence

#     def save_model(self, filepath="cnn_rnn_model_params.pt"):
#         """Saves the PyTorch state dictionary and hyperparameters."""
#         if self.model is None:
#             raise ValueError("No model parameters to save. Fit the model first.")
            
#         save_dict = {
#             'state_dict': self.model.state_dict(),
#             'hyperparams': {
#                 'num_filters': self.num_filters,
#                 'rnn_hidden': self.rnn_hidden,
#                 'n_classes': self.n_classes,
#                 'seq_len': self.seq_len,
#                 'n_inputs': self.n_inputs
#             }
#         }
#         torch.save(save_dict, filepath)
#         print(f"CNN-RNN parameters saved to {filepath}")

#     def load_model(self, filepath="cnn_rnn_model_params.pt"):
#         """Loads weights and topology from a saved file."""
#         data = torch.load(filepath, map_location=self.device)
        
#         h_params = data['hyperparams']
#         self.num_filters = h_params['num_filters']
#         self.rnn_hidden = h_params['rnn_hidden']
#         self.n_classes = h_params['n_classes']
#         self.seq_len = h_params['seq_len']
#         self.n_inputs = h_params['n_inputs']
        
#         self._initialize_model()
#         self.model.load_state_dict(data['state_dict'])
#         self.model.eval()
        
#         print(f"\n[SUCCESS] CNN-RNN Model loaded from: '{filepath}'")
#         print("-" * 40)
#         print("💡 Hyperparameters:")
#         print(f"   • CNN Filters:    {self.num_filters}")
#         print(f"   • GRU Hidden Dim: {self.rnn_hidden}")
#         print(f"   • Target Classes: {self.n_classes}")
#         print(f"   • Sequence Length:{self.seq_len} steps")
#         print(f"   • Input Channels: {self.n_inputs} (F/T axes)")
#         print("-" * 40 + "\n")
        
#     def get_model_size(self):
#         """Calculates and returns the total number of trainable parameters in the CNN-RNN model."""
#         if self.model is None:
#             raise ValueError("Model is not initialized. Call fit() or load_model() first.")
            
#         # Count all parameters that require gradients (are trainable)
#         total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
#         # Format the output nicely
#         if total_params >= 1_000_000:
#             formatted_size = f"{total_params / 1_000_000:.2f}M"
#         elif total_params >= 1_000:
#             formatted_size = f"{total_params / 1_000:.2f}K"
#         else:
#             formatted_size = str(total_params)
            
#         print("\n⚙️ CNN-RNN Parameter Breakdown:")
#         print(f"   • Base Architecture: CNN Filters = {self.num_filters}, GRU Hidden = {self.rnn_hidden}")
#         print(f"   • Total Trainable Parameters: {total_params} ({formatted_size})\n")
        
#         return total_params