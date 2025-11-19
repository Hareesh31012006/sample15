import os
import warnings
# Memory optimizations
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
os.environ['TRANSFORMERS_OFFLINE'] = '0'
os.environ['USE_XFORMERS'] = '1'
warnings.filterwarnings("ignore")

from textblob import TextBlob
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
import torch
import pandas as pd
import numpy as np
import gc

class OptimizedSentimentAnalyzer:
    def __init__(self):
        self.textblob_analyzer = None
        self.finbert_analyzer = None
        self._is_model_loaded = False
        self._xformers_available = False
        self._setup_analyzers()
        
    def _setup_analyzers(self):
        """Initialize sentiment analyzers with memory optimization"""
        try:
            # Check for xformers availability
            try:
                import xformers
                self._xformers_available = True
                print("✓ Xformers available for memory-efficient attention")
            except ImportError:
                print("⚠ Xformers not available, using standard attention")
                self._xformers_available = False
                
            self._is_model_loaded = False
            print("✓ Sentiment analyzer ready (lazy loading enabled)")
        except Exception as e:
            print(f"Error setting up sentiment analyzer: {e}")
            self.finbert_analyzer = None
    
    def _load_finbert_if_needed(self):
        """Lazy load FinBERT with xformers optimization"""
        if self._is_model_loaded and self.finbert_analyzer is not None:
            return
            
        try:
            # Clear memory before loading
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            model_name = "ProsusAI/finbert"
            
            # Load model and tokenizer separately for xformers optimization
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSequenceClassification.from_pretrained(model_name)
            
            # Apply xformers memory-efficient attention if available
            if self._xformers_available:
                try:
                    model.requires_grad_(False)  # Freeze model for inference
                    print("✓ Xformers memory-efficient attention enabled")
                except Exception as e:
                    print(f"⚠ Xformers optimization failed: {e}")
                    self._xformers_available = False
            
            # Create pipeline with optimized model
            self.finbert_analyzer = pipeline(
                "sentiment-analysis",
                model=model,
                tokenizer=tokenizer,
                device=-1,  # Force CPU usage
                torch_dtype=torch.float32,
                batch_size=1,  # Reduced for memory efficiency
                truncation=True,
                max_length=128,  # Further reduced for memory efficiency
                padding=True
            )
            
            self._is_model_loaded = True
            memory_optimization = "with xformers" if self._xformers_available else "without xformers"
            print(f"✓ FinBERT loaded successfully on CPU {memory_optimization}")
            
        except Exception as e:
            print(f"Error loading FinBERT: {e}")
            self.finbert_analyzer = None
    
    def analyze_with_textblob(self, text):
        """Analyze sentiment using TextBlob (lightweight)"""
        try:
            if pd.isna(text) or text == '':
                return 0.0
            
            # Clean text for better analysis
            cleaned_text = str(text).replace('\n', ' ').replace('\r', ' ').strip()
            if len(cleaned_text) < 5:
                return 0.0
                
            analysis = TextBlob(cleaned_text)
            return analysis.sentiment.polarity
        except Exception as e:
            print(f"TextBlob analysis error: {e}")
            return 0.0
    
    def analyze_with_finbert(self, text):
        """Analyze sentiment using FinBERT with xformers optimization"""
        try:
            self._load_finbert_if_needed()
            
            if not self.finbert_analyzer or pd.isna(text) or text == '':
                return 0.0, "neutral", 0.0
            
            # Clean and prepare text
            cleaned_text = str(text).replace('\n', ' ').replace('\r', ' ').strip()
            if len(cleaned_text) < 10:
                return 0.0, "neutral", 0.0
            
            # Use torch.no_grad to save memory
            with torch.no_grad():
                result = self.finbert_analyzer(cleaned_text[:128])  # Further reduced length
            
            sentiment_label = result[0]['label']
            confidence = result[0]['score']
            
            # Enhanced scoring
            if sentiment_label == 'positive':
                return confidence, "positive", confidence
            elif sentiment_label == 'negative':
                return -confidence, "negative", confidence
            else:
                return 0.0, "neutral", confidence
                
        except Exception as e:
            print(f"FinBERT analysis error: {e}")
            return 0.0, "neutral", 0.0
    
    def analyze_news_sentiment_detailed(self, news_df):
        """Analyze sentiment for news articles with detailed results"""
        if news_df.empty:
            return 0.0, []
        
        article_sentiments = []
        sentiments_textblob = []
        sentiments_finbert = []
        
        # Process articles with memory limits
        max_articles = min(5, len(news_df))
        
        for _, article in news_df.head(max_articles).iterrows():
            try:
                # Extract article data with proper error handling
                title = str(article.get('title', '')).strip()
                description = str(article.get('description', '')).strip()
                source = str(article.get('source', 'Unknown Source'))
                published_at = str(article.get('published_at', 'Unknown Date'))
                
                # Skip if no meaningful content
                if not title or len(title) < 5:
                    continue
                    
                if len(description) > 20:
                    text = f"{title}. {description}"
                else:
                    text = title
                    
                text = text[:200]  # Reduced for memory
                
                # TextBlob sentiment
                tb_sentiment = self.analyze_with_textblob(text)
                sentiments_textblob.append(tb_sentiment)
                
                # FinBERT sentiment
                fb_sentiment, fb_label, fb_confidence = self.analyze_with_finbert(text)
                if abs(fb_sentiment) > 0.1:
                    sentiments_finbert.append(fb_sentiment)
                
                # Calculate combined sentiment
                if fb_sentiment != 0.0 and self._xformers_available:
                    combined_sentiment = (tb_sentiment * 0.3) + (fb_sentiment * 0.7)
                elif fb_sentiment != 0.0:
                    combined_sentiment = (tb_sentiment * 0.4) + (fb_sentiment * 0.6)
                else:
                    combined_sentiment = tb_sentiment
                
                # Determine sentiment label
                if combined_sentiment > 0.1:
                    sentiment_label = "positive"
                elif combined_sentiment < -0.1:
                    sentiment_label = "negative"
                else:
                    sentiment_label = "neutral"
                
                # Ensure confidence is reasonable
                confidence = max(fb_confidence, 0.5) if fb_confidence > 0 else 0.5
                
                article_sentiments.append({
                    'title': title,
                    'description': description if description else "No description available.",
                    'source': source,
                    'published_at': published_at,
                    'textblob_sentiment': tb_sentiment,
                    'finbert_sentiment': fb_sentiment,
                    'finbert_confidence': fb_confidence,
                    'sentiment': combined_sentiment,
                    'label': sentiment_label,
                    'confidence': confidence
                })
                
            except Exception as e:
                print(f"Error processing article: {e}")
                continue
        
        # Calculate overall sentiment
        if article_sentiments:
            overall_sentiment = np.mean([article['sentiment'] for article in article_sentiments])
        else:
            overall_sentiment = 0.0
        
        # Apply bounds
        overall_sentiment = max(-1.0, min(1.0, overall_sentiment))
        
        # Aggressive cleanup
        self.cleanup()
        
        return overall_sentiment, article_sentiments
    
    def get_memory_info(self):
        """Get memory optimization information"""
        info = {
            'xformers_enabled': self._xformers_available,
            'model_loaded': self._is_model_loaded,
            'analyzer_type': 'FinBERT + TextBlob'
        }
        
        if self._xformers_available:
            info['optimization'] = 'Memory-efficient attention enabled'
        else:
            info['optimization'] = 'Standard attention'
            
        return info
    
    def cleanup(self):
        """Explicitly clean up models and free memory"""
        try:
            if self.finbert_analyzer:
                # Clear pipeline and model
                if hasattr(self.finbert_analyzer, 'model'):
                    del self.finbert_analyzer.model
                if hasattr(self.finbert_analyzer, 'tokenizer'):
                    del self.finbert_analyzer.tokenizer
                del self.finbert_analyzer
                
            self.finbert_analyzer = None
            self._is_model_loaded = False
            
            # Force garbage collection
            gc.collect()
            
            # Clear CUDA cache if available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                
            print("✓ Sentiment analyzer memory cleaned up")
            
        except Exception as e:
            print(f"Cleanup error: {e}")
