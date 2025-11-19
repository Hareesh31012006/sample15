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
        self._setup_analyzers()
        
    def _setup_analyzers(self):
        """Initialize sentiment analyzers with memory optimization"""
        try:
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
                batch_size=2,  # Increased batch size for efficiency
                truncation=True,
                max_length=384  # Balanced length for accuracy vs memory
            )
            self._is_model_loaded = True
            print("✓ FinBERT loaded successfully on CPU")
            
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
            
            # Process with better text cleaning
            cleaned_text = str(text).replace('\n', ' ').replace('\r', ' ').strip()
            if len(cleaned_text) < 10:  # Skip very short texts
                return 0.0
                
            result = self.finbert_analyzer(cleaned_text[:384])  # Increased for better context
            sentiment_label = result[0]['label']
            confidence = result[0]['score']
            
            # Enhanced scoring with confidence weighting
            if sentiment_label == 'positive':
                return confidence * 1.0
            elif sentiment_label == 'negative':
                return confidence * -1.0
            else:
                return 0.0
                
        except Exception as e:
            print(f"FinBERT analysis error: {e}")
            return 0.0
    
    def analyze_news_sentiment(self, news_df):
        """Analyze sentiment for news articles with optimized memory usage"""
        if news_df.empty:
            return 0.0
        
        sentiments_textblob = []
        sentiments_finbert = []
        
        # Process 4 articles for better accuracy (increased from 3)
        max_articles = min(4, len(news_df))
        
        for _, article in news_df.head(max_articles).iterrows():
            # Enhanced text combination for better context
            title = str(article['title']).strip()
            description = str(article.get('description', '')).strip()
            
            # Combine title and description intelligently
            if len(description) > 20:  # Only use substantial descriptions
                text = f"{title}. {description}"
            else:
                text = title
                
            text = text[:400]  # Balanced length
            
            # TextBlob sentiment
            tb_sentiment = self.analyze_with_textblob(text)
            sentiments_textblob.append(tb_sentiment)
            
            # FinBERT sentiment with better error handling
            try:
                fb_sentiment = self.analyze_with_finbert(text)
                # Only include meaningful FinBERT results
                if abs(fb_sentiment) > 0.1:  # Filter out weak signals
                    sentiments_finbert.append(fb_sentiment)
            except Exception as e:
                print(f"FinBERT analysis skipped: {e}")
        
        # Calculate enhanced weighted average
        avg_textblob = np.mean(sentiments_textblob) if sentiments_textblob else 0.0
        
        # Enhanced FinBERT weighting with quality filtering
        if sentiments_finbert:
            avg_finbert = np.mean(sentiments_finbert)
            finbert_confidence = min(len(sentiments_finbert) / max_articles, 1.0)
            # Dynamic weighting based on result quality
            finbert_weight = 0.7 * finbert_confidence  # Up to 70% weight
            textblob_weight = 0.3
            combined_sentiment = (avg_textblob * textblob_weight) + (avg_finbert * finbert_weight)
        else:
            combined_sentiment = avg_textblob
        
        # Apply non-linear scaling for stronger signals
        if abs(combined_sentiment) > 0.3:
            combined_sentiment = combined_sentiment * 1.2  # Amplify strong signals
        elif abs(combined_sentiment) < 0.1:
            combined_sentiment = combined_sentiment * 0.8  # Dampen weak signals
            
        # Ensure within bounds
        combined_sentiment = max(-1.0, min(1.0, combined_sentiment))
        
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
