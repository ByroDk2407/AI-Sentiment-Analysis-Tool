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
            input_dim=8,
            hidden_dim=64,
            num_layers=2,
            output_dim=3
        )
        
        # Prepare data
        X, y = model.prepare_data(df)
        if X is None or y is None:
            raise ValueError("Failed to prepare data")
        
        # Verify data shapes
        logger.info(f"Input shape: {X.shape}, Output shape: {y.shape}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Convert to tensors
        X_train = torch.FloatTensor(X_train)
        y_train = torch.LongTensor(y_train)
        X_test = torch.FloatTensor(X_test)
        y_test = torch.LongTensor(y_test)
        
        # Training parameters
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        num_epochs = 50
        batch_size = 32
        
        # Training loop
        logger.info("Starting training...")
        best_accuracy = 0.0
        
        for epoch in range(num_epochs):
            model.train()
            total_loss = 0
            batch_count = 0
            
            # Train in batches
            for i in range(0, len(X_train), batch_size):
                batch_X = X_train[i:i+batch_size]
                batch_y = y_train[i:i+batch_size]
                
                # Forward pass
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                
                # Check for NaN loss
                if torch.isnan(loss):
                    logger.error(f"NaN loss detected at epoch {epoch+1}, batch {batch_count}")
                    logger.error(f"Outputs: min={outputs.min()}, max={outputs.max()}")
                    raise ValueError("NaN loss detected")
                
                # Backward pass and optimize
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Add gradient clipping
                optimizer.step()
                
                total_loss += loss.item()
                batch_count += 1
            
            # Validation
            model.eval()
            with torch.no_grad():
                test_outputs = model(X_test)
                test_loss = criterion(test_outputs, y_test)
                predictions = torch.argmax(test_outputs, dim=1)
                accuracy = (predictions == y_test).float().mean()
                
                # Save best model
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    os.makedirs('utils/models', exist_ok=True)
                    torch.save(model.state_dict(), 'utils/models/pretrained_lstm.pth')
            
            if (epoch + 1) % 5 == 0:
                logger.info(f'Epoch [{epoch+1}/{num_epochs}], '
                          f'Loss: {total_loss/batch_count:.4f}, '
                          f'Test Loss: {test_loss:.4f}, '
                          f'Accuracy: {accuracy:.4f}')
        
        logger.info(f"Best accuracy achieved: {best_accuracy:.4f}")
        return True
        
    except Exception as e:
        logger.error(f"Error training model: {str(e)}")
        return False

if __name__ == "__main__":
    train_model() 