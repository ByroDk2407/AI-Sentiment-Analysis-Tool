from flask import Flask, jsonify, request
import pandas as pd
from utils.lstm_model import LSTMPredictor
import logging

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize model
model = None

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from request
        data = request.get_json()
        
        # Load data into DataFrame
        df = pd.DataFrame(data['data'])
        
        # Prepare data
        X, _ = model.prepare_data(df)
        
        # Generate predictions
        predictions = model.predict(X)
        
        return jsonify({
            'success': True,
            'predictions': predictions
        })
        
    except Exception as e:
        logger.error(f"Error processing prediction request: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy'})

def init_model():
    """Initialize the LSTM model."""
    global model
    try:
        # Initialize model with appropriate dimensions
        model = LSTMPredictor(
            input_dim=10,  # Adjust based on your features
            hidden_dim=64,
            num_layers=2,
            output_dim=1
        )
        
        # Load pretrained weights
        success = model.load_pretrained('models/pretrained_lstm.pth')
        if not success:
            logger.error("Failed to load pretrained model")
            
    except Exception as e:
        logger.error(f"Error initializing model: {str(e)}")

if __name__ == '__main__':
    init_model()
    app.run(debug=True) 