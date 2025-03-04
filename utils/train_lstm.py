import torch
import torch.nn as nn
import pandas as pd
from lstm_model import LSTMPredictor
import logging
from sklearn.model_selection import train_test_split
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_model():
    try:
        # Load dataset
        df = pd.read_csv('utils/datasets/lstm_dataset.csv')
        logger.info(f"Loaded dataset with {len(df)} records")
        
        # Check for NaN values
        if df.isna().any().any():
            logger.warning("Dataset contains NaN values. Filling with forward/backward fill")
            df = df.fillna(method='ffill').fillna(method='bfill')
        
        # Initialize model
        model = LSTMPredictor(
            input_dim=8,    # Number of features
            hidden_dim=64,  # Hidden layer size
            num_layers=2    # Number of LSTM layers
        )
        
        # Prepare data
        X, y = model.prepare_data(df)
        if X is None or y is None:
            raise ValueError("Failed to prepare data")
        
        # Convert sentiment scores to class probabilities
        sentiment_scores = y[:, 0]  # First column is sentiment
        sentiment_classes = torch.zeros((len(sentiment_scores), 3))  # 3 classes
        
        # Convert scores to class probabilities
        for i, score in enumerate(sentiment_scores):
            if score < -0.2:
                sentiment_classes[i] = torch.tensor([0.8, 0.1, 0.1])  # Negative
            elif score > 0.2:
                sentiment_classes[i] = torch.tensor([0.1, 0.1, 0.8])  # Positive
            else:
                sentiment_classes[i] = torch.tensor([0.1, 0.8, 0.1])  # Neutral
        
        # Prepare targets
        price_targets = torch.FloatTensor(y[:, 1])  # Second column is price
        rate_targets = torch.FloatTensor(y[:, 2])   # Third column is interest rate
        
        # Verify data shapes
        logger.info(f"Input shape: {X.shape}")
        logger.info(f"Target shapes - Sentiment: {sentiment_classes.shape}, Price: {price_targets.shape}, Rate: {rate_targets.shape}")
        
        # Split data
        train_size = int(0.8 * len(X))
        X_train = torch.FloatTensor(X[:train_size])
        X_test = torch.FloatTensor(X[train_size:])
        
        sentiment_train = sentiment_classes[:train_size]
        sentiment_test = sentiment_classes[train_size:]
        
        price_train = price_targets[:train_size]
        price_test = price_targets[train_size:]
        
        rate_train = rate_targets[:train_size]
        rate_test = rate_targets[train_size:]
        
        # Training parameters
        criterion = {
            'sentiment': nn.CrossEntropyLoss(),
            'price': nn.MSELoss(),
            'interest_rate': nn.MSELoss()
        }
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        num_epochs = 50
        batch_size = 32
        
        # Training loop
        logger.info("Starting training...")
        best_loss = float('inf')
        
        for epoch in range(num_epochs):
            model.train()
            total_loss = 0
            batch_count = 0
            
            # Train in batches
            for i in range(0, len(X_train), batch_size):
                batch_X = X_train[i:i+batch_size]
                batch_sentiment = sentiment_train[i:i+batch_size]
                batch_price = price_train[i:i+batch_size]
                batch_rate = rate_train[i:i+batch_size]
                
                # Forward pass
                outputs = model(batch_X)
                
                # Calculate losses
                sentiment_loss = criterion['sentiment'](outputs['sentiment'], batch_sentiment)
                price_loss = criterion['price'](outputs['price'], batch_price.unsqueeze(-1))
                rate_loss = criterion['interest_rate'](outputs['interest_rate'], batch_rate.unsqueeze(-1))
                
                # Combined loss
                loss = sentiment_loss + 0.5 * (price_loss + rate_loss)
                
                # Backward pass and optimize
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                total_loss += loss.item()
                batch_count += 1
            
            # Validation
            model.eval()
            with torch.no_grad():
                test_outputs = model(X_test)
                test_sentiment_loss = criterion['sentiment'](test_outputs['sentiment'], sentiment_test)
                test_price_loss = criterion['price'](test_outputs['price'], price_test.unsqueeze(-1))
                test_rate_loss = criterion['interest_rate'](test_outputs['interest_rate'], rate_test.unsqueeze(-1))
                
                test_loss = test_sentiment_loss + 0.5 * (test_price_loss + test_rate_loss)
                
                # Save best model
                if test_loss < best_loss:
                    best_loss = test_loss
                    torch.save(model.state_dict(), 'utils/models/pretrained_lstm.pth')
            
            if (epoch + 1) % 5 == 0:
                logger.info(f'Epoch [{epoch+1}/{num_epochs}], '
                          f'Loss: {total_loss/batch_count:.4f}, '
                          f'Test Loss: {test_loss:.4f}')
        
        logger.info(f"Best test loss achieved: {best_loss:.4f}")
        return True
        
    except Exception as e:
        logger.error(f"Error training model: {str(e)}")
        return False

if __name__ == "__main__":
    train_model() 