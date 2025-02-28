from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime, timedelta
from utils.config import Config
import logging
from typing import Dict, List
import pandas as pd

logger = logging.getLogger(__name__)

Base = declarative_base()

class Article(Base):
    __tablename__ = 'articles'
    
    id = Column(Integer, primary_key=True)
    title = Column(String)
    content = Column(Text)
    url = Column(String, unique=True)
    source = Column(String)
    date_collected = Column(DateTime, default=datetime.now)
    date_of_article = Column(DateTime, nullable=True)
    sentiment = Column(String)
    confidence = Column(Float)

class DatabaseManager:
    def __init__(self):
        """Initialize database connection."""
        try:
            self.engine = create_engine(Config.DATABASE_URL)
            Base.metadata.create_all(self.engine)
            self.Session = sessionmaker(bind=self.engine)
            logger.info("Database connection initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing database: {str(e)}")
            raise

    def save_article(self, article: Dict, sentiment_scores: Dict) -> bool:
        """Save an article and its sentiment scores to the database."""
        try:
            session = self.Session()
            
            # Clean and normalize the URL
            url = article.get('url', '')
            if '&ved=' in url:
                url = url.split('&ved=')[0]
            if '&usg=' in url:
                url = url.split('&usg=')[0]
            
            # Check if article exists
            existing = session.query(Article).filter(
                Article.url == url
            ).first()
            
            if existing:
                logger.debug(f"Article already exists: {url}")
                session.close()
                return False
            
            # Parse article date - keep original timestamp if available
            article_date = None
            if article.get('date'):
                try:
                    # Remove any timezone info to match database datetime
                    article_date = pd.to_datetime(article['date']).replace(tzinfo=None)
                    
                    # Validate date is not in the future
                    if article_date > datetime.now():
                        logger.warning(f"Article date {article_date} is in future, using current time")
                        article_date = datetime.now()
                        
                    # Validate date is not too old (e.g., more than 2 years)
                    elif article_date < datetime.now() - timedelta(days=730):
                        logger.warning(f"Article date {article_date} is too old, using current time")
                        article_date = datetime.now()
                        
                except Exception as e:
                    logger.warning(f"Could not parse date '{article.get('date')}': {str(e)}")
                    article_date = datetime.now()
            else:
                article_date = datetime.now()
            
            logger.debug(f"Using article date: {article_date} for URL: {url}")
            
            # Create new article
            new_article = Article(
                title=article.get('title', ''),
                content=article.get('content', ''),
                url=url,
                source=article.get('source', ''),
                date_collected=datetime.now(),
                date_of_article=article_date,
                sentiment=sentiment_scores.get('sentiment', 'neutral'),
                confidence=float(sentiment_scores.get('confidence', 0.0))
            )
            
            session.add(new_article)
            session.commit()
            session.close()
            
            logger.debug(f"Successfully saved article: {url}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving article to database: {str(e)}")
            if session:
                session.rollback()
                session.close()
            return False

    def get_recent_sentiments(self, days: int = 30, limit: int = 10000) -> List[Dict]:
        """Get sentiment data from the last N days."""
        try:
            session = self.Session()
            cutoff_date = datetime.now() - timedelta(days=days)
            
            logger.debug(f"Fetching articles since {cutoff_date}")
            
            # Query using date_of_article instead of date_collected
            articles = session.query(Article).filter(
                Article.date_of_article >= cutoff_date
            ).order_by(
                Article.date_of_article.desc()
            ).all()
            
            logger.info(f"Found {len(articles)} articles in database for {days} day period")
            
            results = []
            for article in articles:
                results.append({
                    'title': article.title,
                    'content': article.content,
                    'url': article.url,
                    'source': article.source,
                    'sentiment': article.sentiment,
                    'confidence': article.confidence,
                    'date_collected': article.date_collected.isoformat(),
                    'date_of_article': article.date_of_article.isoformat() if article.date_of_article else None
                })
            
            session.close()
            return results
            
        except Exception as e:
            logger.error(f"Error getting recent sentiments: {str(e)}")
            if session:
                session.close()
            return []

    def reset_database(self):
        """Reset the database by dropping and recreating all tables."""
        try:
            Base.metadata.drop_all(self.engine)
            Base.metadata.create_all(self.engine)
            logger.info("Database reset successful")
            return True
        except Exception as e:
            logger.error(f"Error resetting database: {str(e)}")
            return False

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    # Print the database URL being used
    #print(f"Database URL: {Config.DATABASE_URL}")
    # Test database connection
    print("Testing database connection...")
    try:
        db_manager = DatabaseManager()
        print("Successfully connected to database!")
        
        # Test saving an article
        test_article = {
            'title': 'Test Article',
            'content': 'This is a test article content',
            'source': 'test',
            'url': 'http://test.com'
        }
        test_sentiment = {
            'sentiment': 'positive',
            'confidence': 0.8
        }
        
        db_manager.save_article(test_article, test_sentiment)
        print("Successfully saved test article!")
        
        # Test retrieving articles
        recent = db_manager.get_recent_sentiments(1)
        if recent:
            print("Successfully retrieved recent article!")
            print(f"Title: {recent[0]['title']}")
            print(f"Sentiment: {recent[0]['sentiment']}")
    
    except Exception as e:
        print(f"Database test failed: {str(e)}") 