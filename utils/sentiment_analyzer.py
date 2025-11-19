import os
import warnings
# Suppress xformers warning
os.environ['XFORMERS_DISABLED'] = '1'
warnings.filterwarnings("ignore")

from textblob import TextBlob
from transformers import pipeline
import pandas as pd
import numpy as np

class SentimentAnalyzer:
    def __init__(self):
        self.textblob_analyzer = None
        self.finbert_analyzer = None
        self._setup_analyzers()
    
    def _setup_analyzers(self):
        """Initialize sentiment analyzers with error handling"""
        try:
            # Initialize FinBERT (financial sentiment analysis model)
            self.finbert_analyzer = pipeline(
                "sentiment-analysis",
                model="ProsusAI/finbert",
                tokenizer="ProsusAI/finbert"
            )
        except Exception as e:
            print(f"Error setting up FinBERT: {e}")
            self.finbert_analyzer = None
    
    def analyze_news_sentiment(self, news_df):
        """Analyze sentiment for news articles with error handling"""
        if news_df.empty:
            return 0.0
        
        try:
            sentiments_textblob = []
            
            for _, article in news_df.iterrows():
                # Combine title and description for analysis
                text = f"{article['title']}. {article.get('description', '')}"
                
                # TextBlob sentiment (fallback)
                tb_sentiment = self.analyze_with_textblob(text)
                sentiments_textblob.append(tb_sentiment)
            
            # Use only TextBlob for reliability
            avg_sentiment = np.mean(sentiments_textblob) if sentiments_textblob else 0.0
            
            return avg_sentiment
            
        except Exception as e:
            print(f"Sentiment analysis error: {e}")
            return 0.0
    
    def analyze_with_textblob(self, text):
        """Analyze sentiment using TextBlob only (more reliable)"""
        try:
            if pd.isna(text) or text == '':
                return 0.0
            
            analysis = TextBlob(str(text))
            return analysis.sentiment.polarity
        except Exception as e:
            print(f"TextBlob analysis error: {e}")
            return 0.0
