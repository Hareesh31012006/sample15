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
        # Add this line to ensure setup is called
        self._setup_analyzers()
        
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
        """Lazy load FinBERT only when needed with better memory management"""
        if self._is_model_loaded and self.finbert_analyzer is not None:
            return
            
        try:
            # Clear memory before loading
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Load with CPU only to save memory
            self.finbert_analyzer = pipeline(
                "sentiment-analysis",
                model="ProsusAI/finbert",
                tokenizer="ProsusAI/finbert",
                device=-1,  # Force CPU usage to save memory
                torch_dtype=torch.float32,
                batch_size=1,
                truncation=True,
                max_length=512  # Increased back to 512 for better accuracy
            )
            self._is_model_loaded = True
            print("✓ FinBERT loaded successfully on CPU")
            
        except Exception as e:
            print(f"Error loading FinBERT: {e}")
            # Fallback to TextBlob only
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
            result = self.finbert_analyzer(str(text[:400]))  # Increased text length for accuracy
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
        """Analyze sentiment for news articles with optimized memory usage"""
        if news_df.empty:
            return 0.0
        
        sentiments_textblob = []
        sentiments_finbert = []
        
        # Process 3 articles for better accuracy (increased from 2)
        max_articles = min(3, len(news_df))
        
        for _, article in news_df.head(max_articles).iterrows():
            # Use both title and description for better accuracy
            text = f"{article['title']}. {article.get('description', '')}"
            text = text[:400]  # Limit length but keep more context
            
            # TextBlob sentiment (always available)
            tb_sentiment = self.analyze_with_textblob(text)
            sentiments_textblob.append(tb_sentiment)
            
            # FinBERT sentiment with error handling
            try:
                fb_sentiment = self.analyze_with_finbert(text)
                sentiments_finbert.append(fb_sentiment)
            except Exception as e:
                print(f"FinBERT analysis skipped: {e}")
                sentiments_finbert.append(0.0)
        
        # Calculate weighted average with FinBERT having more weight
        avg_textblob = np.mean(sentiments_textblob) if sentiments_textblob else 0.0
        
        # Use FinBERT only if we have valid results
        valid_finbert_scores = [s for s in sentiments_finbert if s != 0.0]
        if valid_finbert_scores:
            avg_finbert = np.mean(valid_finbert_scores)
            # Weighted average: FinBERT (60%), TextBlob (40%)
            combined_sentiment = (avg_textblob * 0.4) + (avg_finbert * 0.6)
        else:
            combined_sentiment = avg_textblob
        
        # Immediate cleanup after processing
        self.cleanup()
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
