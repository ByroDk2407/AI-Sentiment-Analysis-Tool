import torch
from transformers import BertTokenizer, BertForSequenceClassification
import numpy as np
from typing import Dict, Union
from backend.config import Config

class SentimentAnalyzer:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = BertTokenizer.from_pretrained(Config.BERT_MODEL_NAME)
        self.model = BertForSequenceClassification.from_pretrained(
            Config.BERT_MODEL_NAME,
            num_labels=3  # positive, neutral, negative
        ).to(self.device)
        
    def analyze_text(self, text: str) -> Dict[str, Union[str, float]]:
        """Analyze the sentiment of a given text."""
        # Tokenize and prepare input
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=Config.MAX_LENGTH,
            truncation=True,
            padding=True
        ).to(self.device)
        
        # Get model prediction
        with torch.no_grad():
            outputs = self.model(**inputs)
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=1)
            
        # Convert probabilities to numpy for easier handling
        probs = probabilities.cpu().numpy()[0]
        
        # Determine sentiment label
        sentiment_label = ['negative', 'neutral', 'positive'][np.argmax(probs)]
        
        return {
            'sentiment': sentiment_label,
            'confidence': float(np.max(probs)),
            'probabilities': {
                'negative': float(probs[0]),
                'neutral': float(probs[1]),
                'positive': float(probs[2])
            }
        }
    
    def analyze_batch(self, texts: list) -> list:
        """Analyze sentiment for a batch of texts."""
        return [self.analyze_text(text) for text in texts] 