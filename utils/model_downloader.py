import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import logging
from utils.lstm_model import LSTMPredictor
import os

logger = logging.getLogger(__name__)

def download_and_setup_models():
    """Download and setup both sentiment classification and LSTM models."""
    try:
        # Create models directory if it doesn't exist
        os.makedirs('models', exist_ok=True)
        
        # Download FinBERT for sentiment classification (3 classes)
        sentiment_model = AutoModelForSequenceClassification.from_pretrained(
            "ProsusAI/finbert",
            num_labels=3  # Keep original 3-class classification
        )
        tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
        
        # Save sentiment classification model
        sentiment_model.save_pretrained('models/finbert_sentiment')
        tokenizer.save_pretrained('models/finbert_sentiment')
        logger.info("Successfully saved sentiment classification model")
        
        # Initialize LSTM model with new parameters
        lstm_model = LSTMPredictor(
            input_dim=8,    # Number of features in your dataset
            hidden_dim=64,  # Size of hidden layer
            num_layers=2    # Number of LSTM layers
        )
        
        # Initialize LSTM weights properly
        def init_weights(m):
            if isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight_ih' in name:
                        torch.nn.init.xavier_uniform_(param.data)
                    elif 'weight_hh' in name:
                        torch.nn.init.orthogonal_(param.data)
                    elif 'bias' in name:
                        param.data.fill_(0)
            elif isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0)
        
        # Apply weight initialization
        lstm_model.apply(init_weights)
        
        # Save LSTM model
        torch.save(lstm_model.state_dict(), 'models/pretrained_lstm.pth')
        logger.info("Successfully saved LSTM model weights")
        
        # Verify both models
        test_sentiment = AutoModelForSequenceClassification.from_pretrained('models/finbert_sentiment')
        test_lstm = LSTMPredictor(input_dim=8, hidden_dim=64, num_layers=2)
        test_lstm.load_state_dict(torch.load('models/pretrained_lstm.pth'))
        
        logger.info("Successfully verified both models")
        return True
        
    except Exception as e:
        logger.error(f"Error setting up models: {str(e)}")
        return False

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    success = download_and_setup_models()
    if success:
        print("Successfully downloaded and setup all models")
    else:
        print("Failed to setup models") 