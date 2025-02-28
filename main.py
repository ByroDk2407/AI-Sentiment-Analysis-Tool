import logging
from datetime import datetime
from typing import Dict, List, Optional
import pandas as pd
import plotly.graph_objects as go
import argparse

from utils.social_media_scraper import SocialMediaAggregator
from utils.preprocessor import DataPreprocessor
from utils.db_manager import DatabaseManager
from utils.visualizer import DataVisualizer
from utils.config import Config

# Configure logging to show only warnings and errors by default
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)

# Only show info logs for the main application
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
            raw_data = self.social_media_aggregator.gather_all_data()
            
            collection_stats = {
                source: len(data) for source, data in raw_data.items()
            }
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
                # Debug logging
                logger.debug(f"Attempting to save item: {item.get('url')}")
                logger.debug(f"Sentiment scores: confidence={item.get('confidence')}, sentiment={item.get('sentiment')}")
                
                sentiment_scores = {
                    'confidence': item.get('confidence', 0.0),
                    'sentiment': item.get('sentiment', 'neutral')
                }
                if self.db_manager.save_article(item, sentiment_scores):
                    saved_count += 1
                    logger.debug(f"Successfully saved item {saved_count}")
            
            logger.info(f"Successfully saved {saved_count} items to database")
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
                    'overall_sentiment': overall_sentiment,
                    'sentiment_distribution': sentiment_counts.to_dict(),
                    'total_articles': len(df),
                    'sources_distribution': df['source'].value_counts().to_dict(),
                    'date_range': {
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


if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Real Estate Sentiment Analysis Data Collection')
    parser.add_argument('--collect', action='store_true', help='Collect new data')
    parser.add_argument('--visualise', action='store_true', help='Generate visualizations')
    parser.add_argument('--display', action='store_true', help='Display sample articles from each source and time period')
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
    
    if args.visualise:
        print("\nGenerating visualisations...")
        analyzer.generate_visualizations()
        
    if args.display:
        print("\nDisplaying sample articles...")
        for period in Config.NEWS_API_TIME_PERIODS:
            print(f"\nArticles from last {period} days:")
            articles = analyzer.db_manager.get_recent_sentiments(days=period)
            
            # Filter for NewsAPI articles
            newsapi_articles = [a for a in articles if a['source'] == 'newsapi']
            
            print(f"\nNewsAPI Articles")
            print(f"Found {len(newsapi_articles)} articles")
            
            for article in newsapi_articles:
                print(f"\nTitle: {article['title']}")
                print(f"Date: {article['date_of_article']}")
                print(f"Sentiment: {article['sentiment']} (confidence: {article['confidence']:.2f})")
                print(f"URL: {article['url']}")