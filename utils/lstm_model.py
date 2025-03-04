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
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int):
        super(LSTMPredictor, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # LSTM layer
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        
        # Multiple prediction heads
        self.sentiment_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 3),  # 3 classes for sentiment
            nn.Softmax(dim=1)
        )
        
        self.price_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1)  # Single value for price prediction
        )
        
        self.interest_rate_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1)  # Single value for interest rate prediction
        )
        
        # Initialize scalers
        self.feature_scaler = MinMaxScaler()
        self.target_scaler = MinMaxScaler()
        
        # Initialize metrics
        self.train_loss = 0.0
        self.eval_loss = 0.0
        self.accuracy = 0.0
        self.rate_train_loss = 0.0
        self.rate_eval_loss = 0.0
        self.rate_accuracy = 0.0
        
    def forward(self, x):
        # Initialize hidden state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        
        # LSTM forward pass
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        
        # Use only the last timestep for predictions
        lstm_out = out[:, -1, :]  # Shape: [batch_size, hidden_dim]
        
        # Generate predictions
        sentiment_pred = self.sentiment_head(lstm_out)  # Shape: [batch_size, 3]
        price_pred = self.price_head(lstm_out)         # Shape: [batch_size, 1]
        interest_pred = self.interest_rate_head(lstm_out)  # Shape: [batch_size, 1]
        
        return {
            'sentiment': sentiment_pred,                # Shape: [batch_size, 3]
            'price': price_pred.squeeze(-1),           # Shape: [batch_size]
            'interest_rate': interest_pred.squeeze(-1)  # Shape: [batch_size]
        }
        
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
            self.feature_scaler.fit(feature_data)
            scaled_features = self.feature_scaler.transform(feature_data)
            
            # Create sequences
            X, y = [], []
            for i in range(len(df) - sequence_length):
                X.append(scaled_features[i:(i + sequence_length)])
                # Use the next value after the sequence as target
                next_idx = i + sequence_length
                y.append([
                    df['sentiment_score'].iloc[next_idx],  # Sentiment target
                    df['price'].iloc[next_idx],           # Price target
                    df['Interest_Rate'].iloc[next_idx]    # Interest rate target
                ])
            
            X = np.array(X)
            y = np.array(y)
            
            # Verify data is valid
            if np.isnan(X).any() or np.isnan(y).any():
                logger.error("NaN values found in prepared data")
                return None, None
            
            logger.info(f"Prepared data shapes - X: {X.shape}, y: {y.shape}")
            
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
                
                # Get predictions
                sentiment_probs = predictions['sentiment'].numpy()  # Shape: [batch_size, 3]
                price_preds = predictions['price'].numpy()         # Shape: [batch_size]
                rate_preds = predictions['interest_rate'].numpy()  # Shape: [batch_size]
                
                # Create dummy arrays for other features
                dummy_features = np.zeros((len(price_preds), 8))  # 8 is the number of features
                
                # Fill in the predictions at their respective positions
                dummy_features[:, 2] = rate_preds     # Interest_Rate at index 2
                dummy_features[:, 3] = price_preds    # Price at index 3
                
                # Inverse transform the entire feature set
                unscaled_features = self.feature_scaler.inverse_transform(dummy_features)
                
                # Extract the predictions
                price_preds_scaled = unscaled_features[:, 3]    # Price column
                rate_preds_scaled = unscaled_features[:, 2]     # Interest Rate column
                
                # Verify predictions are valid
                if np.isnan(price_preds_scaled).any() or np.isnan(rate_preds_scaled).any():
                    logger.error("NaN values found in scaled predictions")
                    return None
                    
                # Log shapes for debugging
                logger.info(f"Prediction shapes - "
                           f"sentiment: {sentiment_probs.shape}, "
                           f"price: {price_preds_scaled.shape}, "
                           f"rate: {rate_preds_scaled.shape}")
                
                return {
                    'sentiment': sentiment_probs,
                    'price': price_preds_scaled,
                    'interest_rate': rate_preds_scaled
                }
                
        except Exception as e:
            logger.error(f"Error generating predictions: {str(e)}")
            return None

    def load_pretrained(self, model_path: str) -> bool:
        """Load pretrained model weights."""
        try:
            self.load_state_dict(torch.load(model_path))
            self.eval()
            return True
        except Exception as e:
            logger.error(f"Error loading pretrained model: {str(e)}")
            return False 