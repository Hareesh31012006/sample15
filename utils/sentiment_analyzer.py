import os
import warnings
# Memory optimizations
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
os.environ['TRANSFORMERS_OFFLINE'] = '0'
warnings.filterwarnings("ignore")

from textblob import TextBlob
from transformers import pipeline
import torch
import pandas as pd
import numpy as np
import gc

class OptimizedSentimentAnalyzer:
    def __init__(self):
        self.textblob_analyzer = None
        self.finbert_analyzer = None
        self._is_model_loaded = False
        
    def _setup_analyzers(self):
        """Initialize sentiment analyzers with memory optimization"""
        try:
            # Use a smaller, faster model for initial setup
            # FinBERT is already efficient, but we'll load it only when needed
            self._is_model_loaded = False
            print("✓ Sentiment analyzer ready (lazy loading enabled)")
            
        except Exception as e:
            print(f"Error setting up sentiment analyzer: {e}")
            self.finbert_analyzer = None
    
    def _load_finbert_if_needed(self):
        """Lazy load FinBERT only when needed"""
        if self._is_model_loaded and self.finbert_analyzer is not None:
            return
            
        try:
            # Clear memory before loading
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Load with optimizations - remove device parameter
            self.finbert_analyzer = pipeline(
                "sentiment-analysis",
                model="ProsusAI/finbert",
                tokenizer="ProsusAI/finbert",
                # Remove device=-1 to let transformers auto-detect
                torch_dtype=torch.float32,
                batch_size=1,
                truncation=True,
                max_length=256  # Reduced for memory
            )
            self._is_model_loaded = True
            print("✓ FinBERT loaded successfully")
            
        except Exception as e:
            print(f"Error loading FinBERT: {e}")
            self.finbert_analyzer = None
    
    def analyze_with_textblob(self, text):
        """Analyze sentiment using TextBlob (lightweight)"""
        try:
            if pd.isna(text) or text == '':
                return 0.0
            
            analysis = TextBlob(str(text))
            return analysis.sentiment.polarity
        except Exception as e:
            print(f"TextBlob analysis error: {e}")
            return 0.0
    
    def analyze_with_finbert(self, text):
        """Analyze sentiment using FinBERT with memory optimization"""
        try:
            self._load_finbert_if_needed()
            
            if not self.finbert_analyzer or pd.isna(text) or text == '':
                return 0.0
            
            # Process in chunks to save memory
            result = self.finbert_analyzer(str(text[:256]))  # Reduced text length
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
        finally:
            # Force garbage collection after each analysis
            gc.collect()
    
    def analyze_news_sentiment(self, news_df):
        """Analyze sentiment for news articles with memory optimization"""
        if news_df.empty:
            return 0.0
        
        sentiments_textblob = []
        sentiments_finbert = []
        
        # Limit the number of articles to process
        max_articles = min(3, len(news_df))  # Process max 3 articles
        
        for _, article in news_df.head(max_articles).iterrows():
            # Use only title for analysis (reduces memory)
            text = article['title']
            
            # TextBlob sentiment (always available)
            tb_sentiment = self.analyze_with_textblob(text)
            sentiments_textblob.append(tb_sentiment)
            
            # FinBERT sentiment (only if model loaded successfully)
            fb_sentiment = self.analyze_with_finbert(text)
            sentiments_finbert.append(fb_sentiment)
        
        # Calculate average sentiments
        avg_textblob = np.mean(sentiments_textblob) if sentiments_textblob else 0.0
        
        # Only use FinBERT if we have valid results
        valid_finbert_scores = [s for s in sentiments_finbert if s != 0.0]
        avg_finbert = np.mean(valid_finbert_scores) if valid_finbert_scores else 0.0
        
        # Combined sentiment (weighted average)
        if avg_finbert != 0.0:
            combined_sentiment = (avg_textblob + avg_finbert * 2) / 3
        else:
            combined_sentiment = avg_textblob
        
        # Clear memory after processing
        gc.collect()
        
        return combined_sentiment
    
    def cleanup(self):
        """Explicitly clean up models and free memory"""
        if self.finbert_analyzer:
            del self.finbert_analyzer
        self.finbert_analyzer = None
        self._is_model_loaded = False
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
