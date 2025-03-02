from flask import Flask, render_template, jsonify, request
from utils.db_manager import DatabaseManager
from utils.visualizer import DataVisualizer
from utils.lstm_model import LSTMPredictor
import pandas as pd
from datetime import datetime, timedelta
import logging
import torch

app = Flask(__name__, static_folder='static')
db_manager = DatabaseManager()
visualizer = DataVisualizer()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize LSTM model
model = None

def init_model():
    """Initialize the LSTM model."""
    global model
    try:
        model = LSTMPredictor(
            input_dim=10,
            hidden_dim=64,
            num_layers=2,
            output_dim=1
        )
        
        # Load pretrained weights
        model.load_state_dict(torch.load('models/pretrained_lstm.pth'))
        model.eval()  # Set to evaluation mode
        logger.info("Successfully loaded LSTM model")
        
    except Exception as e:
        logger.error(f"Error initializing model: {str(e)}")

# Initialize model on startup
init_model()

@app.route('/')
def index():
    """Render the main dashboard page."""
    return render_template('index.html')

@app.route('/api/predict', methods=['POST'])
def predict():
    """API endpoint for LSTM predictions."""
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

@app.route('/api/debug')
def debug_data():
    """Debug endpoint to check raw data from database."""
    try:
        data = db_manager.get_recent_sentiments(limit=1000)
        return jsonify({
            'data_count': len(data),
            'sample_data': data[:5] if data else [],
            'columns': list(data[0].keys()) if data else []
        })
    except Exception as e:
        logger.error(f"Debug endpoint error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/data')
def get_data():
    """API endpoint to get sentiment analysis data with time range filter."""
    try:
        days = int(request.args.get('days', 30))
        logger.info(f"Fetching data for last {days} days")
        
        data = db_manager.get_recent_sentiments(days=days, limit=10000)
        
        if not data:
            return jsonify({'error': 'No data available'})
        
        df = pd.DataFrame(data)
        
        # Convert date strings to datetime objects
        df['date_of_article'] = pd.to_datetime(df['date_of_article'], format='mixed')
        df['date_collected'] = pd.to_datetime(df['date_collected'], format='mixed')
        
        # Filter by article date
        cutoff_date = datetime.now() - timedelta(days=days)
        df = df[df['date_of_article'] >= cutoff_date]
        logger.info(f"Filtered to {len(df)} records after date filtering")
        
        # Calculate distributions
        sentiment_dist = df['sentiment'].value_counts().to_dict()
        source_dist = df['source'].value_counts().to_dict()
        
        # Group by date for timeline
        df['date'] = df['date_of_article'].dt.date
        daily_sentiments = df.groupby(['date', 'sentiment']).size().unstack(fill_value=0)
        daily_sentiments = daily_sentiments.sort_index()
        
        response = {
            'total_articles': len(df),
            'overall_sentiment': df['sentiment'].mode()[0] if not df.empty else 'unknown',
            'sentiment_distribution': sentiment_dist,
            'source_distribution': source_dist,
            'timeline_data': {
                'dates': [d.strftime('%Y-%m-%d') for d in daily_sentiments.index],
                'positive': daily_sentiments.get('positive', pd.Series([0] * len(daily_sentiments))).tolist(),
                'negative': daily_sentiments.get('negative', pd.Series([0] * len(daily_sentiments))).tolist(),
                'neutral': daily_sentiments.get('neutral', pd.Series([0] * len(daily_sentiments))).tolist()
            },
            'date_range': {
                'start': df['date_of_article'].min().strftime('%Y-%m-%d') if not df.empty else None,
                'end': df['date_of_article'].max().strftime('%Y-%m-%d') if not df.empty else None
            },
            'articles': data
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error getting data: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/summary')
def get_summary():
    """API endpoint to get summary statistics."""
    data = db_manager.get_recent_sentiments(limit=100)
    if not data:
        return jsonify({'error': 'No data available'})
    
    df = pd.DataFrame(data)
    
    summary = {
        'total_articles': len(df),
        'sentiment_distribution': df['sentiment'].value_counts().to_dict(),
        'source_distribution': df['source'].value_counts().to_dict(),
        'date_range': {
            'start': df['date_collected'].min().isoformat(),
            'end': df['date_collected'].max().isoformat()
        }
    }
    return jsonify(summary)

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({'status': 'healthy'})

@app.route('/api/predict')
def get_predictions():
    """API endpoint for LSTM predictions."""
    try:
        # Load the combined dataset
        df = pd.read_csv('datasets/lstm_dataset.csv')
        
        # Prepare data for LSTM
        X, y = model.prepare_data(df)
        
        # Generate predictions
        predictions = model.predict(X)
        
        # Get dates for the prediction period
        dates = df.index[-len(predictions):].strftime('%Y-%m-%d').tolist()
        actual_prices = df['price_index'].values[-len(predictions):].tolist()
        
        return jsonify({
            'success': True,
            'lstm_data': {
                'dates': dates,
                'actual_prices': actual_prices,
                'predicted_prices': predictions.tolist()
            }
        })
        
    except Exception as e:
        logger.error(f"Error generating predictions: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        })

if __name__ == '__main__':
    app.run(debug=True) 