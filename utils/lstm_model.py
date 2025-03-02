import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple, List
import logging

logger = logging.getLogger(__name__)

class LSTMPredictor(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, output_dim: int):
        super(LSTMPredictor, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # LSTM layer
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        
        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, output_dim)
        
        # Initialize scalers
        self.feature_scaler = MinMaxScaler()
        self.target_scaler = MinMaxScaler()
        
    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        
        # Initialize cell state
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        
        # LSTM forward pass
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        
        # Reshape output for linear layer
        out = self.fc(out[:, -1, :])
        return out
        
    def prepare_data(self, df: pd.DataFrame, sequence_length: int = 30) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for LSTM model."""
        try:
            # Separate features and target
            features = df.drop(['date', 'price'], axis=1).values
            target = df['price'].values.reshape(-1, 1)
            
            # Scale features and target
            scaled_features = self.feature_scaler.fit_transform(features)
            scaled_target = self.target_scaler.fit_transform(target)
            
            # Create sequences
            X, y = [], []
            for i in range(len(df) - sequence_length):
                X.append(scaled_features[i:(i + sequence_length)])
                y.append(scaled_target[i + sequence_length])
                
            return np.array(X), np.array(y)
            
        except Exception as e:
            logger.error(f"Error preparing data: {str(e)}")
            return None, None
            
    def predict(self, X: np.ndarray) -> List[float]:
        """Generate predictions."""
        try:
            self.eval()  # Set model to evaluation mode
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X)
                predictions = self(X_tensor)
                
                # Inverse transform predictions
                predictions = predictions.numpy()
                predictions = self.target_scaler.inverse_transform(predictions)
                
                return predictions.flatten().tolist()
                
        except Exception as e:
            logger.error(f"Error generating predictions: {str(e)}")
            return []

    def load_pretrained(self, model_path: str) -> bool:
        """Load pretrained model weights."""
        try:
            self.load_state_dict(torch.load(model_path))
            self.eval()
            return True
        except Exception as e:
            logger.error(f"Error loading pretrained model: {str(e)}")
            return False 