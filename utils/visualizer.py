import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)

class DataVisualizer:
    def __init__(self):
        # Color scheme for consistency across plots
        self.colors = {
            'positive': '#2ecc71',
            'negative': '#e74c3c',
            'neutral': '#95a5a6',
            'background': '#ffffff',
            'text': '#2c3e50'
        }
        
        # Common layout settings
        self.layout_defaults = {
            'font_family': "Arial, sans-serif",
            'plot_bgcolor': self.colors['background'],
            'paper_bgcolor': self.colors['background'],
            'font_color': self.colors['text']
        }

    def create_sentiment_pie(self, sentiment_distribution: Dict[str, int]) -> go.Figure:
        """Create a pie chart of sentiment distribution."""
        try:
            fig = go.Figure(data=[
                go.Pie(
                    labels=list(sentiment_distribution.keys()),
                    values=list(sentiment_distribution.values()),
                    marker_colors=[self.colors[sent] for sent in sentiment_distribution.keys()],
                    hole=0.4
                )
            ])
            
            fig.update_layout(
                title="Sentiment Distribution",
                **self.layout_defaults
            )
            
            return fig
        except Exception as e:
            logger.error(f"Error creating sentiment pie chart: {str(e)}")
            return None

    def create_source_bar(self, source_distribution: Dict[str, int]) -> go.Figure:
        """Create a bar chart of data sources."""
        try:
            fig = go.Figure(data=[
                go.Bar(
                    x=list(source_distribution.keys()),
                    y=list(source_distribution.values()),
                    marker_color=self.colors['neutral']
                )
            ])
            
            fig.update_layout(
                title="Articles by Source",
                xaxis_title="Source",
                yaxis_title="Number of Articles",
                **self.layout_defaults
            )
            
            return fig
        except Exception as e:
            logger.error(f"Error creating source bar chart: {str(e)}")
            return None

    def create_sentiment_timeline(self, data: List[Dict]) -> go.Figure:
        """Create a timeline of sentiment trends."""
        try:
            df = pd.DataFrame(data)
            df['date_collected'] = pd.to_datetime(df['date_collected'])
            
            # Group by date and sentiment
            daily_sentiments = df.groupby([
                df['date_collected'].dt.date,
                'sentiment'
            ]).size().unstack(fill_value=0)
            
            fig = go.Figure()
            
            for sentiment in ['positive', 'negative', 'neutral']:
                if sentiment in daily_sentiments.columns:
                    fig.add_trace(go.Scatter(
                        x=daily_sentiments.index,
                        y=daily_sentiments[sentiment],
                        name=sentiment.capitalize(),
                        mode='lines+markers',
                        line=dict(color=self.colors[sentiment]),
                        stackgroup='one'  # Stacked area plot
                    ))
            
            fig.update_layout(
                title="Sentiment Trends Over Time",
                xaxis_title="Date",
                yaxis_title="Number of Articles",
                **self.layout_defaults
            )
            
            return fig
        except Exception as e:
            logger.error(f"Error creating sentiment timeline: {str(e)}")
            return None

    def create_dashboard_figures(self, data: List[Dict]) -> Dict[str, go.Figure]:
        """Create all figures for the dashboard."""
        try:
            df = pd.DataFrame(data)
            
            # Calculate distributions
            sentiment_dist = df['sentiment'].value_counts().to_dict()
            source_dist = df['source'].value_counts().to_dict()
            
            return {
                'sentiment_pie': self.create_sentiment_pie(sentiment_dist),
                'source_bar': self.create_source_bar(source_dist),
                'sentiment_timeline': self.create_sentiment_timeline(data)
            }
        except Exception as e:
            logger.error(f"Error creating dashboard figures: {str(e)}")
            return {}

def create_prediction_plot(dates: List[str], actual: List[float], predicted: List[float]) -> go.Figure:
    """Create interactive plot comparing actual vs predicted values."""
    fig = go.Figure()
    
    # Add actual values
    fig.add_trace(go.Scatter(
        x=dates,
        y=actual,
        mode='lines',
        name='Actual',
        line=dict(color='blue')
    ))
    
    # Add predicted values
    fig.add_trace(go.Scatter(
        x=dates,
        y=predicted,
        mode='lines',
        name='Predicted',
        line=dict(color='red', dash='dash')
    ))
    
    # Update layout
    fig.update_layout(
        title='Property Price Predictions',
        xaxis_title='Date',
        yaxis_title='Price',
        hovermode='x unified'
    )
    
    return fig

def generate_market_report(df: pd.DataFrame, predictions: List[float]) -> Dict:
    """Generate market analysis report."""
    latest_price = df['price'].iloc[-1]
    pred_price = predictions[-1]
    price_change = ((pred_price - latest_price) / latest_price) * 100
    
    # Calculate trend indicators
    sentiment_trend = df['sentiment_score'].rolling(window=7).mean().iloc[-1]
    volume_trend = df['article_count'].rolling(window=7).mean().iloc[-1]
    
    return {
        'current_price': latest_price,
        'predicted_price': pred_price,
        'price_change_percent': price_change,
        'market_sentiment': 'Positive' if sentiment_trend > 0 else 'Negative',
        'sentiment_strength': abs(sentiment_trend),
        'market_activity': volume_trend,
        'prediction_confidence': 'High' if volume_trend > 10 else 'Medium'
    } 