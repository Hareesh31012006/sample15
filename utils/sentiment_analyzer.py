import os
import warnings
# Memory optimizations
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
os.environ['TRANSFORMERS_OFFLINE'] = '0'
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
        self._memory_efficient = False
        self._setup_analyzers()
        
    def _setup_analyzers(self):
        """Initialize sentiment analyzers with memory optimization"""
        try:
            # Check for memory efficient options
            self._memory_efficient = True
            print("✓ Memory-efficient sentiment analyzer initialized")
            self._is_model_loaded = False
        except Exception as e:
            print(f"Error setting up sentiment analyzer: {e}")
            self.finbert_analyzer = None
    
    def _load_finbert_if_needed(self):
        """Lazy load FinBERT with memory optimization"""
        if self._is_model_loaded and self.finbert_analyzer is not None:
            return
            
        try:
            # Clear memory before loading
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            model_name = "ProsusAI/finbert"
            
            # Load model and tokenizer with memory optimization
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSequenceClassification.from_pretrained(model_name)
            
            # Apply memory optimizations
            model.requires_grad_(False)  # Freeze model for inference
            model.eval()  # Set to evaluation mode
            
            # Create pipeline with optimized model
            self.finbert_analyzer = pipeline(
                "sentiment-analysis",
                model=model,
                tokenizer=tokenizer,
                device=-1,  # Force CPU usage
                torch_dtype=torch.float32,
                batch_size=1,  # Single batch for memory efficiency
                truncation=True,
                max_length=128,  # Reduced sequence length
                padding=True
            )
            
            self._is_model_loaded = True
            print("✓ FinBERT loaded successfully with memory optimizations")
            
        except Exception as e:
            print(f"Error loading FinBERT: {e}")
            # Fallback to smaller model
            try:
                self.finbert_analyzer = pipeline(
                    "sentiment-analysis",
                    device=-1,
                    torch_dtype=torch.float32,
                    batch_size=1,
                    truncation=True,
                    max_length=128
                )
                self._is_model_loaded = True
                print("✓ Fallback sentiment analyzer loaded")
            except Exception as fallback_error:
                print(f"Fallback also failed: {fallback_error}")
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
        """Analyze sentiment using FinBERT with memory optimization"""
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
                result = self.finbert_analyzer(cleaned_text[:128])
            
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
                
                # FinBERT sentiment with memory protection
                fb_sentiment, fb_label, fb_confidence = 0.0, "neutral", 0.0
                try:
                    fb_sentiment, fb_label, fb_confidence = self.analyze_with_finbert(text)
                    if abs(fb_sentiment) > 0.1:  # Filter weak signals
                        sentiments_finbert.append(fb_sentiment)
                except Exception as e:
                    print(f"FinBERT analysis skipped: {e}")
                
                # Calculate combined sentiment
                if fb_sentiment != 0.0:
                    # Weighted average favoring FinBERT when available
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
    
    def analyze_news_sentiment(self, news_df):
        """Backward compatibility method"""
        sentiment_score, article_sentiments = self.analyze_news_sentiment_detailed(news_df)
        return sentiment_score
    
    def get_memory_info(self):
        """Get memory optimization information"""
        info = {
            'memory_efficient': self._memory_efficient,
            'model_loaded': self._is_model_loaded,
            'analyzer_type': 'FinBERT + TextBlob'
        }
        
        if self._memory_efficient:
            info['optimization'] = 'Memory-efficient mode enabled'
        else:
            info['optimization'] = 'Standard mode'
            
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
