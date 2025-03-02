from flask import Flask, render_template, jsonify, request, Response
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
            input_dim=8,  # Match training input features
            hidden_dim=64,
            num_layers=2,
            output_dim=3  # 3 classes: negative, neutral, positive
        )
        
        # Load pretrained weights
        model.load_state_dict(torch.load('utils/models/pretrained_lstm.pth'))
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
        # Load the dataset
        df = pd.read_csv('utils/datasets/lstm_dataset.csv')
        logger.info("Loaded dataset with shape: %s", df.shape)
        
        # Convert date column to datetime
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        
        # Prepare data for LSTM
        X, _ = model.prepare_data(df)
        if X is None:
            raise ValueError("Failed to prepare data")
        logger.info("Prepared data with shape: %s", X.shape)
            
        # Convert to tensor
        X_tensor = torch.FloatTensor(X)
        
        # Generate predictions
        pred_results = model.predict(X_tensor)
        logger.info("Generated predictions: %s sentiments", len(pred_results['sentiments']))
        
        # Get dates for the prediction period
        dates = df.index[-len(pred_results['sentiments']):].strftime('%Y-%m-%d').tolist()
        
        # Generate market reports
        current_report, future_report = generate_market_report(
            pred_results,
            df['sentiment_score'].values,
            dates
        )
        
        response = {
            'success': True,
            'predictions': {
                'dates': dates,
                'sentiments': pred_results['sentiments'],
                'confidence_scores': pred_results['confidence_scores'],
                'actual_sentiment_scores': df['sentiment_score'].values[-len(pred_results['sentiments']):].tolist()
            },
            'market_report': {
                'current': current_report,
                'future': future_report
            }
        }
        logger.info("Returning prediction response with %s dates", len(dates))
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error generating predictions: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        })

def generate_market_report(predictions, sentiment_scores, dates):
    """Generate market analysis report based on LSTM predictions."""
    try:
        # Calculate current market sentiment (last 7 days)
        current_sentiments = predictions['sentiments'][-7:]
        current_scores = sentiment_scores[-7:]
        current_confidence = predictions['confidence_scores'][-7:]
        
        # Calculate future market sentiment (next 7 days in predictions)
        future_sentiments = predictions['sentiments'][-14:-7]
        future_scores = sentiment_scores[-14:-7]
        future_confidence = predictions['confidence_scores'][-14:-7]
        
        # Analyze current market
        current_sentiment_counts = pd.Series(current_sentiments).value_counts()
        dominant_current = current_sentiment_counts.index[0]
        confidence_current = np.mean(current_confidence)
        avg_score_current = np.mean(current_scores)
        
        # Analyze future market
        future_sentiment_counts = pd.Series(future_sentiments).value_counts()
        dominant_future = future_sentiment_counts.index[0]
        confidence_future = np.mean(future_confidence)
        avg_score_future = np.mean(future_scores)
        
        # Generate reports
        current_report = {
            'sentiment': dominant_current,
            'confidence': confidence_current,
            'analysis': generate_analysis_text(
                dominant_current, 
                confidence_current,
                avg_score_current,
                current_sentiment_counts,
                "current"
            )
        }
        
        future_report = {
            'sentiment': dominant_future,
            'confidence': confidence_future,
            'analysis': generate_analysis_text(
                dominant_future,
                confidence_future,
                avg_score_future,
                future_sentiment_counts,
                "future"
            )
        }
        
        return current_report, future_report
        
    except Exception as e:
        logger.error(f"Error generating market report: {str(e)}")
        return None, None

def generate_analysis_text(sentiment, confidence, avg_score, sentiment_counts, timeframe):
    """Generate natural language analysis of market conditions."""
    total_samples = sum(sentiment_counts)
    sentiment_percentages = {k: (v/total_samples)*100 for k,v in sentiment_counts.items()}
    
    if timeframe == "current":
        time_context = "is currently"
        time_phrase = "Recent data shows"
    else:
        time_context = "is expected to be"
        time_phrase = "Analysis indicates"
    
    # Generate main sentiment statement
    report = f"{time_phrase} the market {time_context} {sentiment} "
    report += f"(confidence: {confidence:.1%}). "
    
    # Add sentiment distribution
    report += "The sentiment distribution shows "
    sentiment_phrases = [
        f"{sentiment_percentages[sent]:.1f}% {sent}"
        for sent in sentiment_counts.index
    ]
    report += ", ".join(sentiment_phrases) + ". "
    
    # Add interpretation
    if sentiment == "positive":
        report += "This suggests favorable market conditions "
    elif sentiment == "negative":
        report += "This indicates challenging market conditions "
    else:
        report += "This suggests stable market conditions "
    
    if timeframe == "future":
        report += "in the coming period. "
    else:
        report += "at present. "
    
    # Add confidence interpretation
    if confidence > 0.8:
        report += "The model shows high confidence in this assessment."
    elif confidence > 0.6:
        report += "The model shows moderate confidence in this assessment."
    else:
        report += "The model shows some uncertainty in this assessment."
    
    return report

def get_market_context(current_report: dict, future_report: dict, recent_articles: list) -> str:
    """Generate context for the AI from market reports and articles."""
    context = f"""
Current Market Condition:
{current_report['analysis']}

Future Market Outlook:
{future_report['analysis']}

Recent Market Articles:
"""
    
    # Add 3 most recent articles
    for article in recent_articles[:3]:
        context += f"\n- {article['title']} ({article['date_of_article']})"
        context += f"\n  Sentiment: {article['sentiment']} (Score: {article['sentiment_score']:.2f})"
    
    return context

def create_chat_prompt(user_query: str, context: str) -> str:
    """Create a prompt for the AI model."""
    return f"""You are an AI assistant specialized in analyzing the Australian real estate market. 
You have access to current market data and sentiment analysis.

Only respond to questions about the Australian real estate market. If the question is about any other topic,
politely ask the user to ask about the Australian real estate market instead. Also please remove any thinking or reasoning print statements.

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
            "messages": [{"role": "user", "content": prompt}]
        }
        response = requests.post(url, json=payload)
        
        if response.status_code == 200:
            # Collect all response lines first
            response_lines = list(response.iter_lines(decode_unicode=True))
            
            # Process complete response
            full_response = ""
            for line in response_lines:
                if line:  # Ignore empty lines
                    try:
                        json_data = json.loads(line)
                        if "message" in json_data and "content" in json_data["message"]:
                            full_response += json_data["message"]["content"]
                    except json.JSONDecodeError:
                        logger.error(f"Failed to parse line: {line}")
            
            # Send complete response to frontend
            return jsonify({
                'success': True,
                'response': full_response
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

if __name__ == '__main__':
    app.run(debug=True) 