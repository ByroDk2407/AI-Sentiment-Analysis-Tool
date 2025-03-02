import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple, List
import logging
import torch.nn.functional as F

logger = logging.getLogger(__name__)

class LSTMPredictor(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, output_dim: int):
        super(LSTMPredictor, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # LSTM layer
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        
        # Fully connected layers
        self.fc1 = nn.Linear(hidden_dim, 32)
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(32, output_dim)
        self.softmax = nn.Softmax(dim=1)
        
        # Initialize scalers
        self.feature_scaler = MinMaxScaler()
        self.target_scaler = MinMaxScaler()
        
    def forward(self, x):
        # Initialize hidden state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        
        # LSTM forward pass
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        
        # Pass through fully connected layers
        out = self.fc1(out[:, -1, :])
        out = F.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.softmax(out)
        
        return out
        
    def prepare_data(self, df: pd.DataFrame, sequence_length: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for LSTM model."""
        try:
            # Select features for prediction
            features = [
                'Inflation', 'GDP', 'Interest_Rate', 'price',
                'sentiment_score', 'article_count',
                'positive_ratio', 'negative_ratio'
            ]
            
            # Verify all features exist
            missing_features = [f for f in features if f not in df.columns]
            if missing_features:
                logger.error(f"Missing features: {missing_features}")
                return None, None
            
            # Handle NaN values
            df = df.fillna(method='ffill').fillna(method='bfill')
            
            # Scale the features
            feature_data = df[features].values
            scaled_features = self.feature_scaler.fit_transform(feature_data)
            
            # Create sequences
            X, y = [], []
            for i in range(len(df) - sequence_length):
                X.append(scaled_features[i:(i + sequence_length)])
                # Convert sentiment to class (0: negative, 1: neutral, 2: positive)
                sentiment = df['sentiment_score'].iloc[i + sequence_length]
                if sentiment < -0.2:
                    y_val = 0  # negative
                elif sentiment > 0.2:
                    y_val = 2  # positive
                else:
                    y_val = 1  # neutral
                y.append(y_val)
            
            X = np.array(X)
            y = np.array(y)
            
            # Verify data is valid
            if np.isnan(X).any() or np.isnan(y).any():
                logger.error("NaN values found in prepared data")
                return None, None
            
            logger.info(f"Prepared data shapes - X: {X.shape}, y: {y.shape}")
            logger.info(f"Class distribution: {np.bincount(y)}")
            
            return X, y
            
        except Exception as e:
            logger.error(f"Error preparing data: {str(e)}")
            return None, None
            
    def predict(self, X: torch.Tensor) -> dict:
        """Generate predictions."""
        try:
            self.eval()  # Set model to evaluation mode
            with torch.no_grad():
                # Generate predictions
                predictions = self(X)
                
                # Get predicted classes and confidence scores
                pred_classes = torch.argmax(predictions, dim=1).numpy()
                confidence_scores = predictions.max(dim=1)[0].numpy()
                
                # Convert predictions to sentiment labels
                sentiment_map = {0: 'negative', 1: 'neutral', 2: 'positive'}
                sentiment_predictions = [sentiment_map[p] for p in pred_classes]
                
                return {
                    'sentiments': sentiment_predictions,
                    'confidence_scores': confidence_scores.tolist()
                }
                
        except Exception as e:
            logger.error(f"Error generating predictions: {str(e)}")
            return {
                'sentiments': [],
                'confidence_scores': []
            }

    def load_pretrained(self, model_path: str) -> bool:
        """Load pretrained model weights."""
        try:
            self.load_state_dict(torch.load(model_path))
            self.eval()
            return True
        except Exception as e:
            logger.error(f"Error loading pretrained model: {str(e)}")
            return False 