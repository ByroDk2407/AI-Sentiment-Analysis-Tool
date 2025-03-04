from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text, text, func, case
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
    sentiment_score = Column(Float)

class DailySentiment(Base):
    __tablename__ = 'daily_sentiments'
    
    id = Column(Integer, primary_key=True)
    date = Column(DateTime, unique=True)
    average_score = Column(Float)
    article_count = Column(Integer)
    positive_count = Column(Integer)
    negative_count = Column(Integer)
    neutral_count = Column(Integer)

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

    def calculate_daily_sentiment(self) -> bool:
        """Calculate and store daily sentiment averages."""
        try:
            session = self.Session()
            
            # Get the date range
            latest_date = session.query(func.max(Article.date_of_article)).scalar()
            if not latest_date:
                return False
                
            # Calculate daily averages using sentiment_score for average
            daily_stats = session.query(
                func.date_trunc('day', Article.date_of_article).label('date'),
                func.avg(Article.sentiment_score).label('avg_score'),  # Use sentiment_score for average
                func.count(Article.id).label('count'),
                func.sum(
                    case(
                        (Article.sentiment == 'positive', 1),
                        else_=0
                    )
                ).label('positive_count'),
                func.sum(
                    case(
                        (Article.sentiment == 'negative', 1),
                        else_=0
                    )
                ).label('negative_count'),
                func.sum(
                    case(
                        (Article.sentiment == 'neutral', 1),
                        else_=0
                    )
                ).label('neutral_count')
            ).group_by(
                func.date_trunc('day', Article.date_of_article)
            ).all()
            
            # Update or insert daily sentiments
            for stats in daily_stats:
                daily_sent = session.query(DailySentiment).filter(
                    DailySentiment.date == stats.date
                ).first()
                
                if not daily_sent:
                    daily_sent = DailySentiment(
                        date=stats.date,
                        average_score=float(stats.avg_score) if stats.avg_score is not None else 0.0,  # Handle NULL values
                        article_count=stats.count,
                        positive_count=stats.positive_count,
                        negative_count=stats.negative_count,
                        neutral_count=stats.neutral_count
                    )
                    session.add(daily_sent)
                else:
                    daily_sent.average_score = float(stats.avg_score) if stats.avg_score is not None else 0.0
                    daily_sent.article_count = stats.count
                    daily_sent.positive_count = stats.positive_count
                    daily_sent.negative_count = stats.negative_count
                    daily_sent.neutral_count = stats.neutral_count
            
            session.commit()
            return True
            
        except Exception as e:
            logger.error(f"Error calculating daily sentiment: {str(e)}")
            if session:
                session.rollback()
            return False
        finally:
            if session:
                session.close()

    def save_article(self, article: Dict, sentiment_scores: Dict) -> bool:
        """Save an article and its sentiment scores to the database."""
        try:
            session = self.Session()
            
            print(f"\nSaving Article:")
            print(f"Title: {article.get('title', '')[:50]}...")
            print(f"Sentiment Score from scores dict: {sentiment_scores.get('sentiment_score')}")
            print(f"Raw sentiment_scores dict: {sentiment_scores}")
            
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
            
            # Parse article date
            article_date = None
            if article.get('date'):
                try:
                    article_date = pd.to_datetime(article['date']).replace(tzinfo=None)
                    if article_date > datetime.now():
                        article_date = datetime.now()
                    elif article_date < datetime.now() - timedelta(days=730):
                        article_date = datetime.now()
                except Exception as e:
                    logger.warning(f"Could not parse date '{article.get('date')}': {str(e)}")
                    article_date = datetime.now()
            else:
                article_date = datetime.now()
            
            # Create new article with sentiment score
            new_article = Article(
                title=article.get('title', ''),
                content=article.get('content', ''),
                url=url,
                source=article.get('source', ''),
                date_collected=datetime.now(),
                date_of_article=article_date,
                sentiment=sentiment_scores.get('sentiment', 'neutral'),
                confidence=float(sentiment_scores.get('confidence', 0.0)),
                sentiment_score=float(sentiment_scores.get('sentiment_score', 0.0))
            )
            
            print(f"New article sentiment_score before save: {new_article.sentiment_score}")
            
            session.add(new_article)
            session.commit()
            
            # Verify the saved score
            saved_article = session.query(Article).filter(Article.url == url).first()
            print(f"Saved article sentiment_score after commit: {saved_article.sentiment_score}")
            
            session.close()
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
            
            #logger.info(f"Found {len(articles)} articles in database for {days} day period")
            
            results = []
            for article in articles:
                results.append({
                    'title': article.title,
                    'content': article.content,
                    'url': article.url,
                    'source': article.source,
                    'sentiment': article.sentiment,
                    'confidence': article.confidence,
                    'sentiment_score': article.sentiment_score,
                    'date_collected': article.date_collected.strftime('%Y-%m-%d %H:%M:%S'),
                    'date_of_article': article.date_of_article.strftime('%Y-%m-%d %H:%M:%S') if article.date_of_article else None
                })
                #print(f"Sentiment Score in db_manager: {article.sentiment_score}")
            
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