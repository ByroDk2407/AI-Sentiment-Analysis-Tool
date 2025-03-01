import re
import pandas as pd
from typing import Dict, List, Union, Tuple
from textblob import TextBlob
from datetime import datetime, timedelta
from utils.sentiment_analyzer import TransformerSentimentAnalyzer
import logging

logger = logging.getLogger(__name__)

class DataPreprocessor:
    def __init__(self):
        # Common patterns to clean
        self.url_pattern = re.compile(r'https?://\S+|www\.\S+')
        self.mention_pattern = re.compile(r'@\w+')
        self.hashtag_pattern = re.compile(r'#\w+')
        self.html_pattern = re.compile(r'<.*?>')
        self.special_chars_pattern = re.compile(r'[^\w\s]')
        
        # Keywords for relevance checking
        self.relevant_keywords = {
            'property', 'real estate', 'housing', 'market', 'mortgage',
            'interest rate', 'auction', 'apartment', 'house', 'price',
            'rent', 'buying', 'selling', 'investment', 'australia'
        }

        # Sentiment thresholds
        self.sentiment_thresholds = {
            'positive': 0.3,    # Polarity > 0.3 is considered positive
            'negative': -0.3,   # Polarity < -0.3 is considered negative
            # Between -0.3 and 0.3 is considered neutral
        }
        
        # Minimum subjectivity threshold to consider sentiment
        self.min_subjectivity = 0.3

        # Initialize the transformer-based sentiment analyzer
        self.sentiment_analyzer = TransformerSentimentAnalyzer()

    def clean_text(self, text: str) -> str:
        """Clean text by removing URLs, mentions, special characters etc."""
        if not isinstance(text, str):
            return ""
            
        text = text.lower()
        text = self.url_pattern.sub(' ', text)
        text = self.mention_pattern.sub(' ', text)
        text = self.hashtag_pattern.sub(' ', text)
        text = self.html_pattern.sub(' ', text)
        text = ' '.join(text.split())  # Remove extra whitespace
        return text

    def is_english(self, text: str) -> bool:
        """Check if text is primarily in English."""
        try:
            blob = TextBlob(text)
            lang = blob.detect_language()
            return lang == 'en'
        except:
            # If language detection fails, check if text contains primarily ASCII characters
            ascii_chars = len([c for c in text if ord(c) < 128])
            return ascii_chars / len(text) > 0.8 if text else False

    def is_relevant(self, text: str) -> bool:
        """Check if text is relevant to real estate market."""
        text = text.lower()
        return any(keyword in text for keyword in self.relevant_keywords)

    def remove_duplicates(self, data: List[Dict]) -> List[Dict]:
        """Remove duplicate entries based on content."""
        df = pd.DataFrame(data)
        df['cleaned_content'] = df['content'].apply(self.clean_text)
        df = df.drop_duplicates(subset=['cleaned_content'])
        return df.to_dict('records')

    def validate_date(self, date_str: str) -> bool:
        """Validate date string and ensure it's recent."""
        try:
            if isinstance(date_str, datetime):
                date = date_str
            else:
                date = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
            
            # Check if date is within last 30 days
            delta = datetime.now() - date
            return delta.days <= 30
        except:
            return False

    def analyze_sentiment(self, text: str) -> Dict:
        """Analyze sentiment of text using transformer model."""
        try:
            # Get sentiment from transformer model
            result = self.sentiment_analyzer.analyze_text(text)
            
            return {
                'sentiment': result['sentiment'],
                'confidence': result['confidence'],
                'probabilities': result['probabilities']
            }
        except Exception as e:
            logger.error(f"Error in sentiment analysis: {str(e)}")
            return None

    def get_sentiment_label(self, sentiment: Dict[str, float]) -> str:
        """Convert sentiment scores to label."""
        if sentiment['subjectivity'] < self.min_subjectivity:
            return 'neutral'
        
        polarity = sentiment['polarity']
        if polarity > self.sentiment_thresholds['positive']:
            return 'positive'
        elif polarity < self.sentiment_thresholds['negative']:
            return 'negative'
        return 'neutral'

    def filter_by_sentiment(self, data: List[Dict], 
                          allowed_sentiments: List[str] = None) -> List[Dict]:
        """Filter data based on sentiment."""
        if allowed_sentiments is None:
            allowed_sentiments = ['positive', 'negative', 'neutral']
        
        filtered_data = []
        
        # Get all cleaned content for batch processing
        texts = [item['cleaned_content'] for item in data if 'cleaned_content' in item]
        sentiment_results = self.sentiment_analyzer.analyze_batch(texts)
        
        # Match results back to data items
        for item, sentiment_result in zip(data, sentiment_results):
            if sentiment_result['sentiment'] in allowed_sentiments:
                item.update({
                    'sentiment': sentiment_result['sentiment'],
                    'sentiment_scores': sentiment_result
                })
                filtered_data.append(item)
        
        return filtered_data

    def preprocess_data(self, data: List[Dict], source: str, 
                       sentiment_filter: List[str] = None) -> List[Dict]:
        """Preprocess data from a specific source."""
        try:
            processed_items = []
            original_count = len(data)
            
            for item in data:
                try:
                    # Get the content based on source type
                    if source == 'reddit':
                        # Combine title and selftext for Reddit posts
                        content = f"{item.get('title', '')} {item.get('content', '')}"
                        if not content.strip():  # Skip if no content
                            continue
                    elif source == 'google_news':
                        content = f"{item.get('title', '')} {item.get('content', '')}"
                        if not content.strip():  # Skip if no content
                            continue
                    else:
                        content = item.get('content', '')
                        if not content:  # Skip if no content
                            continue

                    # Clean the content
                    cleaned_content = self.clean_text(content)
                    if not cleaned_content:  # Skip if cleaning results in empty text
                        continue

                    # Check relevance
                    if not self.is_relevant(cleaned_content):
                        continue

                    # Analyze sentiment
                    sentiment_result = self.analyze_sentiment(cleaned_content)
                    if not sentiment_result:
                        continue

                    # Filter by sentiment if specified
                    if sentiment_filter and sentiment_result['sentiment'] not in sentiment_filter:
                        continue

                    # Create processed item
                    processed_item = {
                        'title': item.get('title', ''),
                        'content': cleaned_content,
                        'url': item.get('url', ''),
                        'source': source,
                        'sentiment': sentiment_result['sentiment'],
                        'confidence': sentiment_result['confidence'],
                        'date_collected': datetime.now().isoformat(),
                        'date': item.get('date', datetime.now().isoformat())
                    }
                    processed_items.append(processed_item)

                except Exception as e:
                    logger.error(f"Error processing item: {str(e)}")
                    continue

            # Log processing statistics
            processed_count = len(processed_items)
            removed_count = original_count - processed_count
            removal_rate = removed_count / original_count if original_count > 0 else 0

            logger.info(
                f"Processing stats for {source}: "
                f"{{'original_count': {original_count}, "
                f"'processed_count': {processed_count}, "
                f"'removed_count': {removed_count}, "
                f"'removal_rate': {removal_rate}}}"
            )

            return processed_items

        except Exception as e:
            logger.error(f"Error processing {source} data: {str(e)}")
            return []

    def get_preprocessing_stats(self, original_data: List[Dict], 
                              processed_data: List[Dict]) -> Dict:
        """Get statistics about the preprocessing results."""
        return {
            'original_count': len(original_data),
            'processed_count': len(processed_data),
            'removed_count': len(original_data) - len(processed_data),
            'removal_rate': (len(original_data) - len(processed_data)) / len(original_data)
            if original_data else 0
        }

    def get_sentiment_distribution(self, data: List[Dict]) -> Dict[str, int]:
        """Get distribution of sentiments in processed data."""
        distribution = {'positive': 0, 'negative': 0, 'neutral': 0}
        
        for item in data:
            if 'sentiment' in item:
                distribution[item['sentiment']] += 1
        
        return distribution

if __name__ == "__main__":
    print("=== Testing Data Preprocessor ===\n")
    
    # Create test data with various scenarios
    test_data = [
        {
            'content': 'Australian housing market shows strong growth! #realestate https://example.com',
            'date': datetime.now().isoformat(),
            'source': 'twitter',
            'title': None
        },
        {
            'content': 'Este mercado inmobiliario no está en inglés',  # Non-English content
            'date': datetime.now().isoformat(),
            'source': 'twitter',
            'title': None
        },
        {
            'content': 'Property prices in Sydney continue to rise @realtor #housing',
            'date': datetime.now().isoformat(),
            'source': 'twitter',
            'title': None
        },
        {
            'content': 'This is a non-relevant post about cats and dogs',  # Irrelevant content
            'date': datetime.now().isoformat(),
            'source': 'twitter',
            'title': None
        },
        {
            'content': 'Housing market analysis shows concerning trends',
            'date': (datetime.now() - timedelta(days=60)).isoformat(),  # Old content
            'source': 'twitter',
            'title': None
        },
        {
            'content': 'Property prices in Sydney continue to rise',  # Duplicate content
            'date': datetime.now().isoformat(),
            'source': 'reddit',
            'title': 'Sydney Property Market Update'
        }
    ]

    preprocessor = DataPreprocessor()
    
    # Test 1: Text Cleaning
    print("=== Test 1: Text Cleaning ===")
    for item in test_data[:3]:  # Test first 3 items
        cleaned = preprocessor.clean_text(item['content'])
        print(f"\nOriginal: {item['content']}")
        print(f"Cleaned : {cleaned}")

    # Test 2: Language Detection
    print("\n=== Test 2: Language Detection ===")
    for item in test_data[:3]:
        is_eng = preprocessor.is_english(item['content'])
        print(f"\nText: {item['content'][:50]}...")
        print(f"Is English: {is_eng}")

    # Test 3: Relevance Check
    print("\n=== Test 3: Relevance Check ===")
    for item in test_data:
        is_rel = preprocessor.is_relevant(item['content'])
        print(f"\nText: {item['content'][:50]}...")
        print(f"Is Relevant: {is_rel}")

    # Test 4: Date Validation
    print("\n=== Test 4: Date Validation ===")
    for item in test_data:
        is_recent = preprocessor.validate_date(item['date'])
        print(f"\nDate: {item['date']}")
        print(f"Is Recent: {is_recent}")

    # Test 5: Full Preprocessing Pipeline
    print("\n=== Test 5: Full Preprocessing Pipeline ===")
    processed_data = preprocessor.preprocess_data(test_data, 'twitter')
    print(f"\nOriginal items: {len(test_data)}")
    print(f"Processed items: {len(processed_data)}")
    
    if processed_data:
        print("\nSample processed item:")
        sample = processed_data[0]
        print(f"Content: {sample['cleaned_content']}")
        print(f"Source: {sample['source']}")
        if 'sentiment' in sample:
            print(f"Sentiment: {sample['sentiment']}")

    # Test 6: Sentiment Analysis
    print("\n=== Test 6: Sentiment Analysis ===")
    sentiment_filtered = preprocessor.preprocess_data(
        test_data, 
        'twitter', 
        sentiment_filter=['positive', 'negative']
    )
    
    print("\nSentiment Distribution:")
    distribution = preprocessor.get_sentiment_distribution(sentiment_filtered)
    for sentiment, count in distribution.items():
        print(f"{sentiment}: {count}")

    # Test 7: Preprocessing Stats
    print("\n=== Test 7: Preprocessing Stats ===")
    stats = preprocessor.get_preprocessing_stats(test_data, processed_data)
    print("\nPreprocessing Statistics:")
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"{key}: {value:.2%}")
        else:
            print(f"{key}: {value}")

    print("\n=== Testing Complete ===") 