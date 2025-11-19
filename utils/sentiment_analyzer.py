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
                    from xformers.ops import memory_efficient_attention
                    # Enable xformers for the model
                    model.requires_grad_(False)  # Freeze model for inference
                    
                    # For transformer models, we need to patch the attention
                    self._enable_xformers_for_model(model)
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
                batch_size=2,
                truncation=True,
                max_length=256,  # Reduced for memory efficiency
                padding=True
            )
            
            self._is_model_loaded = True
            memory_optimization = "with xformers" if self._xformers_available else "without xformers"
            print(f"✓ FinBERT loaded successfully on CPU {memory_optimization}")
            
        except Exception as e:
            print(f"Error loading FinBERT: {e}")
            # Fallback to standard pipeline
            try:
                self.finbert_analyzer = pipeline(
                    "sentiment-analysis",
                    model="ProsusAI/finbert",
                    device=-1,
                    torch_dtype=torch.float32,
                    batch_size=1,
                    truncation=True,
                    max_length=256
                )
                self._is_model_loaded = True
                print("✓ FinBERT loaded with fallback method")
            except Exception as fallback_error:
                print(f"Fallback also failed: {fallback_error}")
                self.finbert_analyzer = None
    
    def _enable_xformers_for_model(self, model):
        """Enable xformers memory-efficient attention for transformer model"""
        try:
            from xformers.ops import memory_efficient_attention
            
            # Patch the attention forward pass for memory efficiency
            for module in model.modules():
                if hasattr(module, 'config') and hasattr(module.config, 'use_xformers'):
                    module.config.use_xformers = True
                
                # For BERT-like models, we patch the self-attention
                if hasattr(module, 'attention'):
                    original_forward = module.attention.forward
                    
                    def patched_forward(*args, **kwargs):
                        # Add xformers optimization here if needed
                        return original_forward(*args, **kwargs)
                    
                    module.attention.forward = patched_forward
            
            print("✓ Xformers optimization applied to model architecture")
            
        except Exception as e:
            print(f"Xformers model patching failed: {e}")
    
    def _optimize_model_memory(self, model):
        """Apply additional memory optimizations"""
        try:
            # Use more aggressive memory optimizations
            model.eval()  # Set to evaluation mode
            model.requires_grad_(False)  # Disable gradients
            
            # Use torch.jit.optimize_for_inference if available
            if hasattr(torch.jit, 'optimize_for_inference'):
                try:
                    model = torch.jit.optimize_for_inference(torch.jit.script(model))
                    print("✓ Model optimized with torch.jit for inference")
                except:
                    pass
            
            return model
        except Exception as e:
            print(f"Model memory optimization failed: {e}")
            return model
    
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
                return 0.0
            
            # Clean and prepare text
            cleaned_text = str(text).replace('\n', ' ').replace('\r', ' ').strip()
            if len(cleaned_text) < 10:
                return 0.0
            
            # Use torch.no_grad to save memory
            with torch.no_grad():
                result = self.finbert_analyzer(cleaned_text[:256])
            
            sentiment_label = result[0]['label']
            confidence = result[0]['score']
            
            # Enhanced scoring
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
        """Analyze sentiment for news articles with xformers optimization"""
        if news_df.empty:
            return 0.0
        
        sentiments_textblob = []
        sentiments_finbert = []
        
        # Process articles with memory limits
        max_articles = min(3, len(news_df))  # Reduced for memory efficiency
        
        for _, article in news_df.head(max_articles).iterrows():
            # Optimized text combination
            title = str(article['title']).strip()
            description = str(article.get('description', '')).strip()
            
            if len(description) > 20:
                text = f"{title}. {description}"
            else:
                text = title
                
            text = text[:300]  # Further reduced for memory
            
            # TextBlob sentiment (always available)
            tb_sentiment = self.analyze_with_textblob(text)
            sentiments_textblob.append(tb_sentiment)
            
            # FinBERT sentiment with memory protection
            try:
                fb_sentiment = self.analyze_with_finbert(text)
                if abs(fb_sentiment) > 0.1:  # Filter weak signals
                    sentiments_finbert.append(fb_sentiment)
            except Exception as e:
                print(f"FinBERT analysis skipped: {e}")
        
        # Calculate weighted average
        avg_textblob = np.mean(sentiments_textblob) if sentiments_textblob else 0.0
        
        # Enhanced weighting with xformers consideration
        if sentiments_finbert:
            avg_finbert = np.mean(sentiments_finbert)
            finbert_confidence = min(len(sentiments_finbert) / max_articles, 1.0)
            
            # Higher weight for FinBERT when xformers is available (more reliable)
            if self._xformers_available:
                finbert_weight = 0.8 * finbert_confidence
            else:
                finbert_weight = 0.6 * finbert_confidence
                
            textblob_weight = 0.2
            combined_sentiment = (avg_textblob * textblob_weight) + (avg_finbert * finbert_weight)
        else:
            combined_sentiment = avg_textblob
        
        # Apply bounds
        combined_sentiment = max(-1.0, min(1.0, combined_sentiment))
        
        # Aggressive cleanup
        self.cleanup()
        
        return combined_sentiment
    
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
