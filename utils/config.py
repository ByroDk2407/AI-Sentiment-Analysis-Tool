import os
from dotenv import load_dotenv
import logging

logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class Config:
    # Twitter API Keys
    TWITTER_API_KEY = os.getenv('TWITTER_API_KEY')
    TWITTER_API_SECRET = os.getenv('TWITTER_API_SECRET')
    TWITTER_BEARER_TOKEN = os.getenv('TWITTER_BEARER_TOKEN')
    TWITTER_ACCESS_TOKEN = os.getenv('TWITTER_ACCESS_TOKEN')
    TWITTER_ACCESS_TOKEN_SECRET = os.getenv('TWITTER_ACCESS_TOKEN_SECRET')
    
    # Reddit API Keys
    REDDIT_CLIENT_ID = os.getenv('REDDIT_CLIENT_ID')
    REDDIT_CLIENT_SECRET = os.getenv('REDDIT_CLIENT_SECRET')
    
    # NewsAPI Configuration
    NEWS_API_KEY = os.getenv('NEWS_API_KEY')
    NEWS_API_QUERIES = [
        'australian real estate',
        'australia housing market',
        #'australian property',
        #'australia real estate investment'
    ]
    NEWS_API_SOURCES = 'au'  # Australian news sources
    NEWS_API_LANGUAGE = 'en'
    NEWS_API_TIME_PERIODS = [7, 30]  # Days to collect data for
    
    # Database Configuration
    DEFAULT_DB_URL = 'postgresql://postgres:Isaac456@localhost:5432/real_estate_sentiment'
    DATABASE_URL = os.getenv('DATABASE_URL', DEFAULT_DB_URL)
    
    # News Sources
    NEWS_SOURCES = [
        'https://www.abc.net.au/news/property/',
        'https://www.domain.com.au/news/',
        'https://www.realestate.com.au/news/'
    ]
    
    # Model Configuration
    BERT_MODEL_NAME = 'bert-base-uncased'
    MAX_LENGTH = 512
    BATCH_SIZE = 16
    
    # Sentiment Thresholds
    SENTIMENT_POSITIVE_THRESHOLD = 0.6
    SENTIMENT_NEGATIVE_THRESHOLD = 0.4
    
    # Twitter/X Configuration
    TWITTER_KEYWORDS = [
        'australian real estate',
        'australia property market',
        'australian housing',
        'property prices australia',
        'real estate australia'
    ]
    TWITTER_MAX_RESULTS = 100
    
    # Reddit Configuration
    REDDIT_SUBREDDITS = [
        'AusProperty',
        'AusFinance',
        'AusRealEstate',
        'AusInvest',
        #'PropertyInvesting',
        #'AusPropertyInvesting',
        #'AusPropertyChat',
        
    ]
    REDDIT_POST_LIMIT = 200  # Increased from 100
    
    # Google News Configuration
    GOOGLE_NEWS_QUERIES = [
        'australian real estate',
        'australia housing market',
        'australian property prices',
        'australia real estate investment',
        'australian mortgage rates',
        'australia property market'
    ]
    GOOGLE_NEWS_PERIOD = '7d'  
    GOOGLE_NEWS_PAGES = 1 
    
    # Selenium Configuration
    CHROME_DRIVER_PATH = os.getenv('CHROME_DRIVER_PATH', '/usr/local/bin/chromedriver') 
    
    @classmethod
    def validate_config(cls):
        """Validate critical configuration values."""
        if not cls.DATABASE_URL or cls.DATABASE_URL == 'postgresql://username:password@localhost/real_estate_sentiment':
            logger.error("Invalid DATABASE_URL configuration")
            cls.DATABASE_URL = cls.DEFAULT_DB_URL
            logger.info(f"Using default DATABASE_URL: {cls.DATABASE_URL}")

# Validate configuration on import
Config.validate_config() 