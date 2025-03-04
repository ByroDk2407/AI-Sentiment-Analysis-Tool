# Import necessary libraries
import logging
from datetime import datetime
from typing import Dict, List, Optional
import pandas as pd
import plotly.graph_objects as go
import argparse
import requests

# Import custom modules
from utils.social_media_scraper import SocialMediaAggregator
from utils.preprocessor import DataPreprocessor
from utils.db_manager import DatabaseManager
from utils.visualizer import DataVisualizer, create_prediction_plot, generate_market_report
from utils.config import Config
from utils.data_preparer import DataPreparer
from utils.db_manager import DailySentiment

# Configure logging to show only warnings and errors by default for the main application
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)



class RealEstateSentimentAnalyzer:
    def __init__(self):
        """Initialize the main application components."""
        self.social_media_aggregator = SocialMediaAggregator()
        self.preprocessor = DataPreprocessor()
        self.db_manager = DatabaseManager()
        self.visualizer = DataVisualizer()
        
    def collect_data(self) -> Dict[str, List[Dict]]:
        """Collect data from all sources."""
        logger.info("Starting data collection from all sources...")
        
        try:
            # Collect data from all sources
            raw_data = self.social_media_aggregator.gather_all_data()
            collection_stats = {source: len(data) for source, data in raw_data.items()}
            logger.info(f"Data collection stats: {collection_stats}")
            return raw_data
        
        except Exception as e:
            logger.error(f"Error collecting data: {str(e)}")
            return {}

    def process_data(self, raw_data: Dict[str, List[Dict]]) -> List[Dict]:
        """Process collected data through the preprocessing pipeline."""
        logger.info("Processing collected data...")
        processed_data = []
        
        try:
            # Process each source separately
            for source, items in raw_data.items():
                processed_items = self.preprocessor.preprocess_data(items, source)
                processed_data.extend(processed_items)
                
            print(f"Successfully processed {len(processed_data)} items")
            return processed_data
            
        except Exception as e:
            logger.error(f"Error in data processing: {str(e)}")
            return []

    def store_data(self, processed_data: List[Dict]) -> int:
        """Store processed data in database."""
        logger.info("Storing processed data in database...")
        
        try:
            saved_count = 0
            for item in processed_data:
                # Get sentiment analysis results
                sentiment_scores = {
                    'confidence':       item.get('confidence', 0.0),
                    'sentiment':        item.get('sentiment', 'neutral'),
                    'sentiment_score':  item.get('sentiment_score', 0.0)
                }
                
                if self.db_manager.save_article(item, sentiment_scores):
                    saved_count += 1
            
            return saved_count
            
        except Exception as e:
            logger.error(f"Error storing data: {str(e)}")
            return 0

    def get_sentiment_summary(self, days: int = 7) -> Dict:
        """Get sentiment summary for the specified time period."""
        try:
            logger.info(f"Fetching sentiment data for past {days} days...")
            recent_data = self.db_manager.get_recent_sentiments(days)
            
            if not recent_data:
                logger.warning("No sentiment data found for the specified time period")
                return {}
            
            logger.info(f"Found {len(recent_data)} records")
            
            try:
                df = pd.DataFrame(recent_data)
                
                # Ensure required columns exist
                required_columns = ['sentiment', 'source', 'date_collected']
                missing_columns = [col for col in required_columns if col not in df.columns]
                if missing_columns:
                    logger.error(f"Missing required columns: {missing_columns}")
                    return {}
                
                # Calculate summary statistics
                sentiment_counts = df['sentiment'].value_counts()
                overall_sentiment = sentiment_counts.index[0] if not sentiment_counts.empty else "unknown"
                
                summary = {
                    'overall_sentiment':        overall_sentiment,
                    'sentiment_distribution':   sentiment_counts.to_dict(),
                    'total_articles':           len(df),
                    'sources_distribution':     df['source'].value_counts().to_dict(),
                    'date_range':{              
                                                'start': df['date_collected'].min().isoformat() if not df.empty else None,
                                                'end': df['date_collected'].max().isoformat() if not df.empty else None
                    }
                }
                
                logger.info(f"Successfully generated summary for {len(df)} articles")
                return summary
                
            except Exception as e:
                logger.error(f"Error generating summary: {str(e)}")
                return {}
            
        except Exception as e:
            logger.error(f"Error getting sentiment summary: {str(e)}")
            return {}

    def prepare_lstm_data(self):
        """Prepare data for LSTM model."""
        try:
            # Initialize components
            data_preparer = DataPreparer(self.db_manager)
            
            # Calculate daily sentiments
            self.db_manager.calculate_daily_sentiment()
            
            # Load all datasets
            property_data = data_preparer.load_property_data('utils\datasets\combined_property_data.csv')
            sentiment_data = data_preparer.get_sentiment_data()
            
            # Prepare LSTM dataset with aligned dates
            lstm_data = data_preparer.prepare_lstm_dataset(property_data=property_data,sentiment_data=sentiment_data)
            
            if lstm_data.empty:
                print("Failed to prepare LSTM dataset")
                return None
            
            return lstm_data
            
        except Exception as e:
            print(f"Error preparing LSTM data: {str(e)}")
            return None

    def generate_visualizations(self, data: List[Dict]) -> Dict[str, go.Figure]:
        """Generate all visualizations."""
        try:
            return self.visualizer.create_dashboard_figures(data)
        except Exception as e:
            logger.error(f"Error generating visualizations: {str(e)}")
            return {}

    def run_analysis_pipeline(self) -> Dict:
        """Run the complete analysis pipeline."""
        logger.info("Starting analysis pipeline...")
        
        # Collect data
        raw_data = self.collect_data()
        if not raw_data:
            logger.error("No data collected. Stopping pipeline.")
            return {}
        
        # Process data
        processed_data = self.process_data(raw_data)
        if not processed_data:
            logger.error("No processed data. Stopping pipeline.")
            return {}
        
        # Store data
        if not self.store_data(processed_data):
            logger.error("Failed to store data.")
        
        # Get summary
        summary = self.get_sentiment_summary()
        
        return summary

    def generate_predictions(self):
        """Generate predictions using LSTM model."""
        try:
            # Load LSTM dataset
            df = pd.read_csv('datasets/lstm_dataset.csv')
            
            # Make API request for predictions
            response = requests.post(
                'http://localhost:5000/predict',
                json={'data': df.to_dict('records')}
            )
            
            if response.status_code == 200:
                data = response.json()
                if data['success']:
                    predictions = data['predictions']
                    
                    # Create visualization
                    fig = create_prediction_plot(
                        dates=df['date'].tolist(),
                        actual=df['price'].tolist(),
                        predicted=predictions
                    )
                    
                    # Generate report
                    report = generate_market_report(df, predictions)
                    
                    return {
                        'figure': fig,
                        'report': report
                    }
                    
            return None
            
        except Exception as e:
            logger.error(f"Error generating predictions: {str(e)}")
            return None


if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Real Estate Sentiment Analysis Data Collection')
    parser.add_argument('--collect', action='store_true', help='Collect new data')
    parser.add_argument('--visualise', action='store_true', help='Generate visualizations')
    parser.add_argument('--display', action='store_true', help='Display sample articles from each source and time period')
    parser.add_argument('--combine', action='store_true', help='Combine datasets and prepare LSTM data')
    parser.add_argument('--showsent', action='store_true', help='Display daily sentiment scores')
    parser.add_argument('--setupmodel', action='store_true', help='Download and setup the LSTM model')
    args = parser.parse_args()

    # Initialize the analyzer
    analyzer = RealEstateSentimentAnalyzer()
    
    if args.collect:
        print("Starting data collection...")
        
        # Collect data
        data = analyzer.collect_data()
        total_items = sum(len(items) for items in data.values())
        print(f"\nCollected {total_items} items:")
        for source, items in data.items():
            print(f"{source}: {len(items)} items")
        
        # Process and save data
        processed_data = analyzer.process_data(data)
        print(f"\nProcessed {len(processed_data)} items")
        
        # Save to database
        saved_count = analyzer.store_data(processed_data)
        print(f"\nSaved {saved_count} items to database")
        
        # Calculate daily sentiments after saving articles
        print("\nCalculating daily sentiment averages...")
        if analyzer.db_manager.calculate_daily_sentiment():
            print("Successfully calculated daily sentiment scores")
        else:
            print("Failed to calculate daily sentiment scores")
    
    if args.combine:
        print("\nCombining datasets for LSTM...")
        lstm_data = analyzer.prepare_lstm_data()
        
        if lstm_data is not None and not lstm_data.empty:
            print("\nLSTM Dataset Preview:")
            print("\nShape:", lstm_data.shape)
            print("\nColumns:", lstm_data.columns.tolist())
            print("\nDate Range:")
            print(f"Start: {lstm_data['date'].min()}")
            print(f"End: {lstm_data['date'].max()}")
            
            print("\nFirst few rows:")
            pd.set_option('display.max_columns', None)
            pd.set_option('display.width', None)
            print(lstm_data.head())
            
            print("\nDescriptive Statistics:")
            print(lstm_data.describe())
            
            print("\nMissing Values:")
            missing = lstm_data.isnull().sum()
            if missing.any():
                print(missing[missing > 0])
            else:
                print("No missing values found")
            
            print("\nDataset saved to: datasets/lstm_dataset.csv")
        else:
            print("Failed to create LSTM dataset")
    
    if args.visualise:
        print("\nGenerating visualisations...")
        analyzer.generate_visualizations()
        
    if args.display:
        print("\nDisplaying sample articles...")
        for period in Config.NEWS_API_TIME_PERIODS:
            print(f"\nArticles from last {period} days:")
            articles = analyzer.db_manager.get_recent_sentiments(days=period)
            
            # Group articles by source
            sources = set(a['source'] for a in articles)
            
            for source in sources:
                source_articles = [a for a in articles if a['source'] == source]
                
                print(f"\n{source.upper()} Articles")
                print(f"Found {len(source_articles)} articles")
                
                for article in source_articles:
                    print(f"\nTitle:\t\t {article['title']}")
                    print(f"Date:\t\t {article['date_of_article']}")
                    print(f"Date Collected:\t {article['date_collected']}")
                    print(f"Sentiment:\t {article['sentiment']} (confidence: {article['confidence']:.2f})")
                    print(f"Sentiment Score:\t {article['sentiment_score']:.2f}")
                    print("\n\n")
    
    if args.showsent:
        print("\nDisplaying Daily Sentiment Scores...")
        try:
            session = analyzer.db_manager.Session()
            daily_sentiments = session.query(DailySentiment).order_by(DailySentiment.date.desc()).all()
            
            if not daily_sentiments:
                print("No sentiment data available.")
            else:
                print("\nDaily Sentiment Analysis:")
                print(f"{'Date':<12} {'Score':>8} {'Articles':>8} {'Pos%':>7} {'Neg%':>7} {'Neu%':>7}")
                print("-" * 55)
                
                for sent in daily_sentiments:
                    total = sent.article_count
                    pos_pct = (sent.positive_count / total * 100) if total > 0 else 0
                    neg_pct = (sent.negative_count / total * 100) if total > 0 else 0
                    neu_pct = (sent.neutral_count / total * 100) if total > 0 else 0
                    
                    # Format the date and numbers
                    date_str = sent.date.strftime('%Y-%m-%d')
                    score_str = f"{sent.average_score:>6.5f}"
                    articles_str = f"{sent.article_count:>8}"
                    pos_str = f"{pos_pct:>6.1f}%"
                    neg_str = f"{neg_pct:>6.1f}%"
                    neu_str = f"{neu_pct:>6.1f}%"
                    
                    print(f"{date_str} {score_str} {articles_str} {pos_str} {neg_str} {neu_str}")
                
                # Print summary statistics
                print("\nSummary Statistics:")
                scores = [s.average_score for s in daily_sentiments]
                print(f"Average Sentiment Score: {sum(scores)/len(scores):.2f}")
                print(f"Highest Score: {max(scores):.2f}")
                print(f"Lowest Score: {min(scores):.2f}")
                print(f"Total Articles Analyzed: {sum(s.article_count for s in daily_sentiments)}")
                
            session.close()
            
        except Exception as e:
            print(f"Error displaying sentiment scores: {str(e)}")
            if session:
                session.close()

    if args.setupmodel:
        print("\nSetting up LSTM model...")
        from utils.model_downloader import download_and_setup_models
        if download_and_setup_models():
            print("Successfully set up LSTM model")
        else:
            print("Failed to set up LSTM model")