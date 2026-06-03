import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# ==========================================
# 1. The Core PyTorch Module (Hidden from API)
# ==========================================
class _SignalTransformerNetwork(nn.Module):
    """
    Signal Transformer for Haptic Signals
    Based on 'Learning an Efficient Terrain Representation for Haptic Localization...'
    """
    def __init__(self, num_classes, input_dim=6, seq_len=160, d_model=16, nhead=2, d_ff=8, embed_dim=256, dropout=0.1):
        super(_SignalTransformerNetwork, self).__init__()
        
        # 1. Input Linear Projection (6D -> 16D) [cite: 534]
        self.input_proj = nn.Linear(input_dim, d_model)
        
        # 2. Layer Normalization [cite: 535]
        self.layer_norm = nn.LayerNorm(d_model)
        
        # 3. Learnable Positional Encoding [cite: 535]
        self.pos_embed = nn.Parameter(torch.randn(1, seq_len, d_model))
        
        # 4. Transformer Encoder (h=2, d_m=16, d_ff=8) [cite: 537]
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=d_ff,
            dropout=dropout,
            activation='relu', 
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
        
        # 5. Embedding Generation [cite: 538, 539]
        self.batch_norm = nn.BatchNorm1d(d_model)
        self.embed_fc = nn.Linear(d_model, embed_dim)
        self.relu = nn.ReLU()
        
        # 6. Classification Head (Adapted for your specific benchmark)
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        # x shape: (Batch, Seq_Len, Features) -> (B, 160, 6)
        
        # Projection & Normalization
        x = self.input_proj(x)
        x = self.layer_norm(x)
        
        # Add Positional Encoding
        x = x + self.pos_embed
        
        # Transformer
        x = self.transformer_encoder(x)
        
        # Average pooling over the sequence dimension [cite: 537]
        x = x.mean(dim=1) 
        
        # Embedding Generation (BatchNorm -> Dense -> ReLU) [cite: 538]
        x = self.batch_norm(x)
        latent_embedding = self.relu(self.embed_fc(x))
        
        # Final Classification
        out = self.classifier(latent_embedding)
        return out

# ==========================================
# 2. Your Wrapper Class Format
# ==========================================
class SignalTransformerClassifier:
    def __init__(self, embed_dim=256, epochs=50, batch_size=128, lr=5e-4, random_state=42):
        # Hyperparameters match the paper exactly [cite: 537, 539, 571, 572]
        self.d_model = 16
        self.nhead = 2
        self.d_ff = 8
        self.embed_dim = embed_dim 
        
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.random_state = random_state
        
        self.model = None
        self.n_classes = None
        self.seq_len = None
        self.n_inputs = None
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _initialize_model(self):
        """Sets up the PyTorch model and optimizer."""
        torch.manual_seed(self.random_state)
        self.model = _SignalTransformerNetwork(
            num_classes=self.n_classes,
            input_dim=self.n_inputs,
            seq_len=self.seq_len,
            d_model=self.d_model,
            nhead=self.nhead,
            d_ff=self.d_ff,
            embed_dim=self.embed_dim
        ).to(self.device)
        
        self.criterion = nn.CrossEntropyLoss()
        # The paper specifically used AdamW optimizer with cosine decay [cite: 570, 571]
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=2e-4)

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

        print(f"Training Signal Transformer on {n_samples} gait cycles using {self.device}...")
        
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
                
        print("Signal Transformer Training complete.")

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

    def save_model(self, filepath="signal_transformer_params.pt"):
        """Saves the PyTorch state dictionary and model parameters."""
        if self.model is None:
            raise ValueError("No model parameters to save. Fit the model first.")
            
        save_dict = {
            'state_dict': self.model.state_dict(),
            'hyperparams': {
                'embed_dim': self.embed_dim,
                'n_classes': self.n_classes,
                'seq_len': self.seq_len,
                'n_inputs': self.n_inputs
            }
        }
        torch.save(save_dict, filepath)
        print(f"Signal Transformer parameters saved to {filepath}")

    def load_model(self, filepath="signal_transformer_params.pt"):
        """Loads weights and topology from a saved file."""
        data = torch.load(filepath, map_location=self.device)
        
        h_params = data['hyperparams']
        self.embed_dim = h_params['embed_dim']
        self.n_classes = h_params['n_classes']
        self.seq_len = h_params['seq_len']
        self.n_inputs = h_params['n_inputs']
        
        self._initialize_model()
        self.model.load_state_dict(data['state_dict'])
        self.model.eval()
        
        print(f"\n[SUCCESS] Signal Transformer Model loaded from: '{filepath}'")
        print("-" * 40)
        print("💡 Hyperparameters:")
        print(f"   • Attention Heads: {self.nhead}")
        print(f"   • Embedding Size:  {self.embed_dim}")
        print(f"   • Target Classes:  {self.n_classes}")
        print("-" * 40 + "\n")

    def get_model_size(self):
        """Calculates and returns the total number of trainable parameters in the Signal Transformer model."""
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
            
        print("\n⚙️ Signal Transformer Parameter Breakdown:")
        print(f"   • Architecture: Heads = {self.nhead}, d_model = {self.d_model}, Embed Dim = {self.embed_dim}")
        print(f"   • Total Trainable Parameters: {total_params} ({formatted_size})\n")
        
        return total_params