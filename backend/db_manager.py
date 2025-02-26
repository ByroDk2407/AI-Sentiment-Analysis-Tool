from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
from backend.config import Config

Base = declarative_base()

class Article(Base):
    __tablename__ = 'articles'
    
    id = Column(Integer, primary_key=True)
    title = Column(String(500))
    content = Column(Text)
    source = Column(String(200))
    url = Column(String(1000))
    sentiment = Column(String(20))
    sentiment_score = Column(Float)
    date_collected = Column(DateTime, default=datetime.utcnow)

class DatabaseManager:
    def __init__(self):
        self.engine = create_engine(Config.DATABASE_URL)
        Base.metadata.create_all(self.engine)
        Session = sessionmaker(bind=self.engine)
        self.session = Session()
    
    def save_article(self, article_data: dict, sentiment_data: dict):
        """Save article and its sentiment analysis to database."""
        article = Article(
            title=article_data['title'],
            content=article_data['content'],
            source=article_data['source'],
            url=article_data['url'],
            sentiment=sentiment_data['sentiment'],
            sentiment_score=sentiment_data['confidence']
        )
        
        self.session.add(article)
        self.session.commit()
    
    def get_recent_sentiments(self, limit: int = 100):
        """Get recent sentiment data from database."""
        return self.session.query(Article).order_by(
            Article.date_collected.desc()
        ).limit(limit).all() 