import os
from dotenv import load_dotenv

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
    
    # Database Configuration
    DATABASE_URL = os.getenv('DATABASE_URL', 'postgresql://localhost/real_estate_sentiment')
    
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
        'AusRealEstate'
    ]
    REDDIT_POST_LIMIT = 100 #May change later for longer reddits / deeper analysis.
    
    # Google News Configuration
    GOOGLE_NEWS_QUERY = 'australian real estate OR property market'
    GOOGLE_NEWS_PERIOD = '7d'  # Last 7 days
    
    # Selenium Configuration
    CHROME_DRIVER_PATH = os.getenv('CHROME_DRIVER_PATH', '/usr/local/bin/chromedriver') 