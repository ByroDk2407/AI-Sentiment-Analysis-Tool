import logging
from db_manager import DatabaseManager, Base
from sqlalchemy import create_engine, inspect
from config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def verify_schema(engine):
    """Verify that all expected columns exist in the database."""
    inspector = inspect(engine)
    columns = inspector.get_columns('articles')
    column_names = [col['name'] for col in columns]
    
    expected_columns = [
        'id', 'title', 'content', 'url', 'source', 
        'date_collected', 'date_of_article', 'sentiment', 'confidence'
    ]
    
    logger.info("Current columns in database: %s", column_names)
    
    for col in expected_columns:
        if col not in column_names:
            logger.error(f"Missing column: {col}")
            return False
    return True

def reset_database():
    """Reset and initialize the database with the new schema."""
    try:
        # Create engine
        engine = create_engine(Config.DATABASE_URL)
        
        # Drop all existing tables
        logger.info("Dropping all existing tables...")
        Base.metadata.drop_all(engine)
        
        # Create all tables with new schema
        logger.info("Creating new tables with updated schema...")
        Base.metadata.create_all(engine)
        
        # Verify schema
        if not verify_schema(engine):
            logger.error("Schema verification failed!")
            return False
            
        logger.info("Database reset successful!")
        logger.info("Schema verification passed - all columns present including date_of_article")
        
        # Initialize DatabaseManager to verify connection
        db = DatabaseManager()
        logger.info("Database connection verified")
        
        return True
        
    except Exception as e:
        logger.error(f"Error resetting database: {str(e)}")
        return False

if __name__ == "__main__":
    print("Starting database reset...")
    success = reset_database()
    if success:
        print("Database reset complete. You can now run data collection.")
    else:
        print("Database reset failed. Check the logs for details.") 