from flask import Flask, render_template, jsonify, request, Response, url_for
from utils.db_manager import DatabaseManager
from utils.visualizer import DataVisualizer
from utils.lstm_model import LSTMPredictor
import pandas as pd
from datetime import datetime, timedelta
import logging
import torch
import numpy as np
import requests
from typing import Optional
import json 
from requests.exceptions import RequestException
import os
import re

app = Flask(__name__, 
    static_folder='static',
    static_url_path='/static'
)
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
            input_dim=8,    # Match your dataset features
            hidden_dim=64,
            num_layers=2
        )
        
        # Load pretrained weights
        model.load_state_dict(torch.load('utils/models/pretrained_lstm.pth'))
        model.eval()  # Set to evaluation mode
        logger.info("Successfully loaded LSTM model")
        return True
        
    except Exception as e:
        logger.error(f"Error initializing model: {str(e)}")
        return False

# Initialize model on startup
if not init_model():
    logger.error("Failed to initialize LSTM model")

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

@app.route('/api/predictions')
def get_predictions():
    try:
        if model is None:
            return jsonify({'error': 'LSTM model not initialized'}), 500
            
        # Load the LSTM dataset
        df = pd.read_csv('utils/datasets/lstm_dataset.csv')
        df['date'] = pd.to_datetime(df['date'])
        
        # Generate dates for the next 30 days from the last date in the dataset
        last_date = df['date'].iloc[-1]
        future_dates = pd.date_range(
            start=last_date, 
            periods=30, 
            freq='D'
        )
        
        # Get the last 30 days of actual data
        last_30_days = df.tail(30).copy()
        
        # Prepare data for predictions
        X, _ = model.prepare_data(df)
        if X is None:
            return jsonify({'error': 'Failed to prepare data'}), 500
            
        X_tensor = torch.FloatTensor(X)
        
        # Get predictions
        predictions = model.predict(X_tensor)
        if predictions is None:
            return jsonify({'error': 'Failed to generate predictions'}), 500
        
        # Get the last 30 predictions
        try:
            response_data = {
                'dates': future_dates.strftime('%Y-%m-%d').tolist(),  # Use generated future dates
                'actual_prices': last_30_days['price'].tolist(),
                'predicted_prices': predictions['price'][-30:].tolist(),
                'actual_rates': last_30_days['Interest_Rate'].tolist(),
                'predicted_rates': predictions['interest_rate'][-30:].tolist(),
                'confidence': float(np.mean(np.max(predictions['sentiment'][-30:], axis=1)))
            }
            
            # Log the dates for debugging
            logger.info(f"Date range: {response_data['dates'][0]} to {response_data['dates'][-1]}")
            
            # Validate the data
            for key, value in response_data.items():
                if key != 'dates' and key != 'confidence':
                    value_array = np.array(value)
                    if np.isnan(value_array).any():
                        logger.error(f"NaN values found in {key}")
                        return jsonify({'error': f'Invalid predictions found in {key}'}), 500
                    if len(value_array) != 30:
                        logger.error(f"Incorrect length for {key}: {len(value_array)}")
                        return jsonify({'error': f'Invalid length for {key}'}), 500
            
            return jsonify(response_data)
            
        except Exception as e:
            logger.error(f"Error preparing response: {str(e)}")
            logger.error(f"Response data: {response_data}")
            return jsonify({'error': 'Failed to prepare prediction response'}), 500
            
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({'error': str(e)}), 500

def generate_market_report(predictions, sentiment_scores, dates):
    """Generate current and future market reports based on predictions."""
    try:
        # Get the most recent predictions
        latest_price = predictions['price'][-1]
        latest_rate = predictions['interest_rate'][-1]
        latest_sentiment = np.argmax(predictions['sentiment'][-1])  # Get most likely sentiment class
        
        # Calculate trends (using last 7 predictions)
        price_trend = np.mean(np.diff(predictions['price'][-7:]))
        rate_trend = np.mean(np.diff(predictions['interest_rate'][-7:]))
        
        # Map sentiment index to string
        sentiment_map = {0: 'negative', 1: 'neutral', 2: 'positive'}
        current_sentiment = sentiment_map[latest_sentiment]
        
        # Generate current market report
        current_report = {
            'price_level': latest_price,
            'interest_rate': latest_rate,
            'price_trend': 'increasing' if price_trend > 0 else 'decreasing',
            'rate_trend': 'increasing' if rate_trend > 0 else 'decreasing',
            'sentiment': current_sentiment,
            'confidence': float(np.max(predictions['sentiment'][-1]))
        }
        
        # Calculate future sentiment based on predicted trends
        future_sentiment = 'positive' if price_trend > 0 and rate_trend < 0 else \
                         'negative' if price_trend < 0 and rate_trend > 0 else \
                         'neutral'
        
        # Generate future market report
        future_report = {
            'price_trend': 'increasing' if price_trend > 0 else 'decreasing',
            'rate_trend': 'increasing' if rate_trend > 0 else 'decreasing',
            'price_change_pct': (predictions['price'][-1] / predictions['price'][0] - 1) * 100,
            'rate_change_pct': (predictions['interest_rate'][-1] / predictions['interest_rate'][0] - 1) * 100,
            'confidence': float(np.mean(np.max(predictions['sentiment'], axis=1))),
            'sentiment': future_sentiment  # Add sentiment to future report
        }
        
        return current_report, future_report
        
    except Exception as e:
        logger.error(f"Error generating market report: {str(e)}")
        return None, None

def get_market_context(current_report, future_report, recent_articles):
    """Generate market context for AI responses."""
    try:
        if current_report is None or future_report is None:
            return "Unable to generate market context due to missing reports."
            
        context = "Current Market Analysis:\n"
        context += f"- Current house price level: ${current_report['price_level']:,.2f}\n"
        context += f"- Interest rate: {current_report['interest_rate']:.2f}%\n"
        context += f"- Price trend: {current_report['price_trend']}\n"
        context += f"- Interest rate trend: {current_report['rate_trend']}\n"
        context += f"- Market sentiment: {current_report['sentiment']} (confidence: {current_report['confidence']:.2f})\n"
        
        context += "\nFuture Market Outlook:\n"
        context += f"- Expected price trend: {future_report['price_trend']}\n"
        context += f"- Expected rate trend: {future_report['rate_trend']}\n"
        context += f"- Projected price change: {future_report['price_change_pct']:.1f}%\n"
        context += f"- Projected rate change: {future_report['rate_change_pct']:.1f}%\n"
        context += f"- Forecast confidence: {future_report['confidence']:.2f}\n"
        
        context += "\nRecent Market News:\n"
        for article in recent_articles[:5]:  # Include 5 most recent articles
            context += f"\n- {article['title']} ({article['date_of_article']})"
            context += f"\n  Sentiment: {article['sentiment']} (Score: {article['sentiment_score']:.2f})"
        
        return context
        
    except Exception as e:
        logger.error(f"Error generating market context: {str(e)}")
        return "Error generating market context."

@app.route('/api/market-report')
def get_market_report():
    """API endpoint for market analysis report."""
    try:
        # Get predictions
        df = pd.read_csv('utils/datasets/lstm_dataset.csv')
        df['date'] = pd.to_datetime(df['date'])
        
        X, _ = model.prepare_data(df)
        X_tensor = torch.FloatTensor(X)
        predictions = model.predict(X_tensor)
        
        if predictions is None:
            return jsonify({'error': 'Failed to generate predictions'}), 500
        
        # Get recent articles
        recent_articles = db_manager.get_recent_sentiments(days=7)
        
        # Generate reports
        current_report, future_report = generate_market_report(
            predictions,
            df['sentiment_score'].values,
            df['date'].values
        )
        
        if current_report is None or future_report is None:
            return jsonify({'error': 'Failed to generate market reports'}), 500
        
        return jsonify({
            'current_report': current_report,
            'future_report': future_report
        })
        
    except Exception as e:
        logger.error(f"Error generating market report: {str(e)}")
        return jsonify({'error': str(e)}), 500

def create_chat_prompt(user_query: str, context: str) -> str:
    """Create a prompt for the AI model."""
    return f"""You are an AI assistant specialized in analyzing the Australian real estate market. 
You have access to current market data and sentiment analysis.

Only respond to questions about the Australian real estate market. If the question is about any other topic,
politely ask the user to ask about the Australian real estate market instead. Also please remove all text within the <text> tags.

Base your responses on this current market data:

{context}

When responding:
- Reference specific data points from the market reports
- Cite recent articles when relevant
- Be clear about whether you're discussing current conditions or future outlook
- Express confidence levels when making predictions
- Stay focused on the Australian real estate market

User Question: {user_query}

Assistant Response:"""

@app.route('/api/chat', methods=['POST'])
def chat():
    """API endpoint for the AI chatbot."""
    try:
        data = request.get_json()
        user_query = data.get('message', '')
        
        # Get current market context
        df = pd.read_csv('utils/datasets/lstm_dataset.csv')
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        
        # Get predictions and reports
        X, _ = model.prepare_data(df)
        X_tensor = torch.FloatTensor(X)
        pred_results = model.predict(X_tensor)
        current_report, future_report = generate_market_report(
            pred_results,
            df['sentiment_score'].values,
            df.index
        )
        
        # Get recent articles
        recent_articles = db_manager.get_recent_sentiments(days=7)
        
        # Generate context for AI
        context = get_market_context(current_report, future_report, recent_articles)
        
        # Create full prompt
        prompt = create_chat_prompt(user_query, context)
        
        # Set up the base URL for the local Ollama API
        url = "http://localhost:11434/api/chat"
        
        # Define the payload
        payload = {
            "model": "deepseek-r1:1.5b",
            "messages": [{"role": "user", "content": prompt}],
            "stream": False
        }
        response = requests.post(url, json=payload)
        
        if response.status_code == 200:
            # Get the response content
            response_data = response.json()
            if "message" in response_data and "content" in response_data["message"]:
                # Clean up the response by removing text within <text> tags
                content = response_data["message"]["content"]
                # Remove all text between <text> and </text> tags
                #cleaned_response = re.sub(r'<text>.*?</text>', '', content, flags=re.DOTALL)
                cleaned_content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL)
                # Remove any extra whitespace or newlines
                #cleaned_response = re.sub(r'\n\s*\n', '\n', cleaned_response.strip())
                
                return jsonify({
                    'success': True,
                    'response': cleaned_content
                })
            else:
                return jsonify({
                    'success': False,
                    'error': 'Invalid response format from Ollama'
                })
        else:
            return jsonify({
                'success': False,
                'error': 'Failed to get response from Ollama'
            })
            
    except Exception as e:
        logger.error(f"Chat error: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/api/ollama-health', methods=['GET'])
def check_ollama_health():
    """Check if Ollama is running and responding."""
    try:
        response = requests.get("http://localhost:11434/api/health", timeout=5)
        return jsonify({
            "status": "healthy" if response.status_code == 200 else "unhealthy",
            "code": response.status_code,
            "response": response.text
        })
    except requests.exceptions.RequestException as e:
        return jsonify({
            "status": "unhealthy",
            "error": str(e)
        }), 503

# Add this route to explicitly serve static files
@app.route('/static/<path:filename>')
def serve_static(filename):
    return app.send_static_file(filename)

if __name__ == '__main__':
    app.run(debug=True) 