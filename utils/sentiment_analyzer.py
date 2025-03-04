import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import Dict, List, Union
import numpy as np
from utils.config import Config
import logging

logger = logging.getLogger(__name__)

class TransformerSentimentAnalyzer:
    def __init__(self):
        # Use locally saved FinBERT for sentiment analysis
        self.model_name = "models/finbert_sentiment"
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load tokenizer and model from local directory if exists, otherwise download
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        except:
            logger.info("Local model not found, downloading from HuggingFace...")
            self.tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
            self.model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
        
        self.model.to(self.device)
        
        # Sentiment labels
        self.labels = ['negative', 'neutral', 'positive']
        
        # Confidence threshold for sentiment classification
        self.confidence_threshold = 0.8

    def calculate_sentiment_score(self, probabilities: Dict[str, float]) -> float:
        """Calculate sentiment score from -1 to 1 based on probabilities."""
        # Weight the probabilities to create a score between -1 and 1
        score = (probabilities['positive'] - probabilities['negative'])
        
        # Apply neutral probability as a dampening factor
        neutral_factor = 1 - probabilities['neutral']
        score = score * neutral_factor
        #print(f"Probabilities: {probabilities['positive']}, {probabilities['neutral']}, {probabilities['negative']}")
        #print(f"Neutral Factor: {neutral_factor}")
        #print(f"Score: {score}")
        return float(score)

    def analyze_text(self, text: str) -> Dict[str, Union[str, float]]:
        """Analyze sentiment of a single text."""
        try:
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                probabilities = torch.nn.functional.softmax(outputs.logits, dim=1)
            
            probs = probabilities[0].cpu().numpy()
            prob_dict = {
                label: float(prob)
                for label, prob in zip(self.labels, probs)
            }
            
            max_prob = np.max(probs)
            pred_idx = np.argmax(probs)
            
            # Calculate sentiment score (-1 to 1)
            sentiment_score = self.calculate_sentiment_score(prob_dict)
            #print(f"Sentiment Score: {sentiment_score}")
            if max_prob >= self.confidence_threshold:
                sentiment = self.labels[pred_idx]
            else:
                sentiment = 'neutral'
            
            return {
                'sentiment': sentiment,
                'confidence': float(max_prob),
                'probabilities': prob_dict,
                'sentiment_score': sentiment_score  # Add sentiment score to return value
            }
        except Exception as e:
            logger.error(f"Error in sentiment analysis: {str(e)}")
            return None

    def analyze_batch(self, texts: List[str], batch_size: int = 16) -> List[Dict]:
        """Analyze sentiment for a batch of texts."""
        results = []
        
        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # Tokenize batch
            inputs = self.tokenizer(
                batch_texts,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True
            ).to(self.device)
            
            # Get predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
                probabilities = torch.nn.functional.softmax(outputs.logits, dim=1)
            
            # Process each item in batch
            batch_probs = probabilities.cpu().numpy()
            for probs in batch_probs:
                prob_dict = {
                    label: float(prob)
                    for label, prob in zip(self.labels, probs)
                }
                
                max_prob = np.max(probs)
                pred_idx = np.argmax(probs)
                
                # Calculate sentiment score
                sentiment_score = self.calculate_sentiment_score(prob_dict)
                
                # Determine sentiment label
                if max_prob >= self.confidence_threshold:
                    sentiment = self.labels[pred_idx]
                else:
                    sentiment = 'neutral'
                
                results.append({
                    'sentiment': sentiment,
                    'confidence': float(max_prob),
                    'probabilities': prob_dict,
                    'sentiment_score': sentiment_score
                })
        
        return results


if __name__ == "__main__":
    # Test the sentiment analyzer
    analyzer = TransformerSentimentAnalyzer()
    
    test_texts = [
        "The housing market is showing strong signs of growth with increasing property values.",
        "Real estate prices are crashing due to rising interest rates and market uncertainty.",
        "Property prices remained stable this quarter with normal trading volumes.",
        "Experts predict significant opportunities in the Australian real estate market.",
        "Housing affordability crisis deepens as prices continue to outpace wage growth."
    ]
    
    print("Testing individual text analysis:")
    for text in test_texts:
        result = analyzer.analyze_text(text)
        print(f"\nText: {text}")
        print(f"Sentiment: {result['sentiment']}")
        print(f"Confidence: {result['confidence']:.2f}")
        print("Probabilities:", {k: f"{v:.2f}" for k, v in result['probabilities'].items()})
    
    print("\nTesting batch analysis:")
    batch_results = analyzer.analyze_batch(test_texts)
    for text, result in zip(test_texts, batch_results):
        print(f"\nText: {text}")
        print(f"Sentiment: {result['sentiment']}")
        print(f"Confidence: {result['confidence']:.2f}")
        print(f"Sentiment Score: {result['sentiment_score']:.2f}")