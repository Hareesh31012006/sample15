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
        """Initialize sentiment analyzers"""
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
    
    def analyze_with_textblob(self, text):
        """Analyze sentiment using TextBlob"""
        try:
            if pd.isna(text) or text == '':
                return 0.0
            
            analysis = TextBlob(str(text))
            return analysis.sentiment.polarity
        except Exception as e:
            print(f"TextBlob analysis error: {e}")
            return 0.0
    
    def analyze_with_finbert(self, text):
        """Analyze sentiment using FinBERT"""
        try:
            if not self.finbert_analyzer or pd.isna(text) or text == '':
                return 0.0
            
            result = self.finbert_analyzer(str(text[:512]))  # Limit text length
            sentiment_label = result[0]['label']
            confidence = result[0]['score']
            
            # Convert to numeric score
            if sentiment_label == 'positive':
                return confidence
            elif sentiment_label == 'negative':
                return -confidence
            else:
                return 0.0
                
        except Exception as e:
            print(f"FinBERT analysis error: {e}")
            return 0.0
    
    def analyze_news_sentiment(self, news_df):
        """Analyze sentiment for news articles"""
        if news_df.empty:
            return 0.0
        
        sentiments_textblob = []
        sentiments_finbert = []
        
        for _, article in news_df.iterrows():
            # Combine title and description for analysis
            text = f"{article['title']}. {article.get('description', '')}"
            
            # TextBlob sentiment
            tb_sentiment = self.analyze_with_textblob(text)
            sentiments_textblob.append(tb_sentiment)
            
            # FinBERT sentiment
            fb_sentiment = self.analyze_with_finbert(text)
            sentiments_finbert.append(fb_sentiment)
        
        # Calculate average sentiments
        avg_textblob = np.mean(sentiments_textblob) if sentiments_textblob else 0.0
        avg_finbert = np.mean(sentiments_finbert) if sentiments_finbert else 0.0
        
        # Combined sentiment (weighted average)
        combined_sentiment = (avg_textblob + avg_finbert * 2) / 3
        
        return combined_sentiment
