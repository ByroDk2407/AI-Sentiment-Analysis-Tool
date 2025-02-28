from config import Config
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_database_url():
    """Test that DATABASE_URL is properly configured."""
    print("\nTesting DATABASE_URL configuration:")
    print(f"Current DATABASE_URL: {Config.DATABASE_URL}")
    
    # Check if it contains default values
    if 'username' in Config.DATABASE_URL or 'password' in Config.DATABASE_URL:
        print("WARNING: DATABASE_URL contains default values!")
    else:
        print("DATABASE_URL appears to be properly configured.")

if __name__ == "__main__":
    test_database_url() 