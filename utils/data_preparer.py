import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List
import logging
from utils.db_manager import DailySentiment

logger = logging.getLogger(__name__)

class DataPreparer:
    def __init__(self, db_manager):
        self.db_manager = db_manager
        
    def load_property_data(self, filepath: str) -> pd.DataFrame:
        """Load and preprocess property sales data."""
        try:
            df = pd.read_csv(filepath)
            df['date'] = pd.to_datetime(df['date'])
            return df
        except Exception as e:
            logger.error(f"Error loading property data: {str(e)}")
            return pd.DataFrame()
    
    def load_economic_data(self, filepath: str) -> pd.DataFrame:
        """Load and preprocess economic data."""
        try:
            df = pd.read_csv(filepath)
            df['date'] = pd.to_datetime(df['date'])
            return df
        except Exception as e:
            logger.error(f"Error loading economic data: {str(e)}")
            return pd.DataFrame()
    
    def get_sentiment_data(self) -> pd.DataFrame:
        """Get sentiment data from database."""
        try:
            session = self.db_manager.Session()
            daily_sentiments = session.query(DailySentiment).all()
            
            data = []
            for sent in daily_sentiments:
                data.append({
                    'date': sent.date,
                    'sentiment_score': sent.average_score,
                    'article_count': sent.article_count,
                    'positive_ratio': sent.positive_count / sent.article_count if sent.article_count > 0 else 0,
                    'negative_ratio': sent.negative_count / sent.article_count if sent.article_count > 0 else 0
                })
            
            df = pd.DataFrame(data)
            df['date'] = pd.to_datetime(df['date'])
            return df
            
        except Exception as e:
            logger.error(f"Error getting sentiment data: {str(e)}")
            return pd.DataFrame()
        finally:
            if session:
                session.close()
    
    def prepare_lstm_dataset(self, property_data: pd.DataFrame,  sentiment_data: pd.DataFrame) -> pd.DataFrame: #economic_data: pd.DataFrame,
        """Prepare aligned dataset for LSTM model."""
        try:
            logger.info("Preparing LSTM dataset...")
            
            # Get date ranges
            sentiment_dates = set(sentiment_data['date'].dt.date)
            property_dates = set(property_data['date'].dt.date)
            #economic_dates = set(economic_data['date'].dt.date)
            
            # Find common dates between all datasets
            common_dates = sentiment_dates.intersection(property_dates)#.intersection(economic_dates)
            
            if not common_dates:
                logger.error("No overlapping dates found between datasets")
                return pd.DataFrame()
            
            # Filter each dataset to common dates
            sentiment_filtered = sentiment_data[sentiment_data['date'].dt.date.isin(common_dates)]
            property_filtered = property_data[property_data['date'].dt.date.isin(common_dates)]
            #economic_filtered = economic_data[economic_data['date'].dt.date.isin(common_dates)]
            
            # Merge datasets
            #merged = pd.merge(property_filtered, economic_filtered, on='date', how='inner')
            merged = pd.merge(property_filtered, sentiment_filtered, on='date', how='inner')
            
            # Sort by date
            merged = merged.sort_values('date')
            
            # Save LSTM dataset
            merged.to_csv('datasets/lstm_dataset.csv', index=False)
            
            logger.info(f"Created LSTM dataset with {len(merged)} rows")
            logger.info(f"Date range: {merged['date'].min()} to {merged['date'].max()}")
            
            return merged
            
        except Exception as e:
            logger.error(f"Error preparing LSTM dataset: {str(e)}")
            return pd.DataFrame()
    
    def combine_datasets(self, property_data: pd.DataFrame, sentiment_data: pd.DataFrame) -> pd.DataFrame:
        """Combine property and sentiment data (for general use)."""
        try:
            # Merge datasets on date
            combined = pd.merge(
                property_data,
                sentiment_data,
                on='date',
                how='left'
            )
            
            # Forward fill sentiment scores for missing dates
            combined['sentiment_score'] = combined['sentiment_score'].fillna(method='ffill')
            combined['article_count'] = combined['article_count'].fillna(0)
            combined['positive_ratio'] = combined['positive_ratio'].fillna(method='ffill')
            combined['negative_ratio'] = combined['negative_ratio'].fillna(method='ffill')
            
            # Save combined dataset
            combined.to_csv('datasets/combined_property_sentiment_data.csv', index=False)
            logger.info("Successfully saved combined dataset")
            
            return combined
            
        except Exception as e:
            logger.error(f"Error combining datasets: {str(e)}")
            return pd.DataFrame() 