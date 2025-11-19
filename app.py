import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
import gc
import warnings
import psutil
import sys
warnings.filterwarnings('ignore')

# Import optimized modules
from utils.data_fetcher import DataFetcher
from utils.sentiment_analyzer import OptimizedSentimentAnalyzer
from utils.model import StockPredictor
from utils.visualization import StockVisualizer

# Load environment variables
load_dotenv()

# Configure Streamlit page
st.set_page_config(
    page_title="NextTick - AI Stock Prediction",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

class MemoryOptimizedNextTickApp:
    def __init__(self):
        self.data_fetcher = None
        self.sentiment_analyzer = None
        self.stock_predictor = None
        self.visualizer = None
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize components with memory optimization"""
        try:
            # Initialize components only when needed
            if not hasattr(self, 'data_fetcher') or self.data_fetcher is None:
                self.data_fetcher = DataFetcher()
            
            if not hasattr(self, 'sentiment_analyzer') or self.sentiment_analyzer is None:
                self.sentiment_analyzer = OptimizedSentimentAnalyzer()
            
            if not hasattr(self, 'stock_predictor') or self.stock_predictor is None:
                self.stock_predictor = StockPredictor()
            
            if not hasattr(self, 'visualizer') or self.visualizer is None:
                self.visualizer = StockVisualizer()
                
        except Exception as e:
            st.error(f"Error initializing app: {e}")
    
    def _get_memory_usage(self):
        """Get current memory usage"""
        try:
            process = psutil.Process(os.getpid())
            return process.memory_info().rss / 1024 / 1024  # MB
        except:
            return 0
    
    def _check_memory_threshold(self):
        """Check if memory usage is too high"""
        current_memory = self._get_memory_usage()
        if current_memory > 500:  # 500MB threshold
            st.warning(f"🔄 High memory usage detected ({current_memory:.1f}MB). Clearing cache...")
            self._force_memory_cleanup()
            return True
        return False
    
    def _force_memory_cleanup(self):
        """Forceful memory cleanup"""
        try:
            # Clear sentiment analyzer first (biggest memory user)
            if self.sentiment_analyzer:
                self.sentiment_analyzer.cleanup()
            
            # Clear large datasets from session state
            large_keys = ['stock_data', 'news_data', 'technical_data', 'model_data']
            for key in large_keys:
                if key in st.session_state:
                    del st.session_state[key]
            
            # Clear components
            self.stock_predictor = None
            self.visualizer = None
            
            # Force garbage collection
            gc.collect()
            
            # Reinitialize essential components
            self._initialize_components()
            
            st.success("🧹 Memory cleanup completed!")
            
        except Exception as e:
            st.warning(f"Memory cleanup warning: {e}")
    
    def _smart_memory_management(self):
        """Smart memory management with proactive cleanup"""
        if 'analysis_count' not in st.session_state:
            st.session_state.analysis_count = 0
            st.session_state.last_memory_clear = datetime.now()
            st.session_state.initial_memory = self._get_memory_usage()
        
        st.session_state.analysis_count += 1
        
        # Check memory threshold first
        if self._check_memory_threshold():
            return
        
        # Clear memory every 3 analyses or 3 minutes (more aggressive)
        current_time = datetime.now()
        time_since_clear = (current_time - st.session_state.last_memory_clear).total_seconds()
        
        if st.session_state.analysis_count >= 3 or time_since_clear > 180:  # 3 minutes
            st.session_state.analysis_count = 0
            st.session_state.last_memory_clear = current_time
            self._clear_memory_cache()
    
    def _clear_memory_cache(self):
        """Enhanced memory clearing without losing accuracy"""
        try:
            # Clear sentiment analyzer memory
            if self.sentiment_analyzer:
                self.sentiment_analyzer.cleanup()
            
            # Keep only essential session state
            essential_data = {}
            essential_keys = ['current_cache_key', 'current_symbol', 'analysis_count', 'last_memory_clear']
            
            for key in essential_keys:
                if key in st.session_state:
                    essential_data[key] = st.session_state[key]
            
            # Clear all session state
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            
            # Restore essential data only
            for key, value in essential_data.items():
                st.session_state[key] = value
            
            # Force garbage collection
            gc.collect()
            
            # Show memory usage info
            current_memory = self._get_memory_usage()
            if 'initial_memory' in st.session_state:
                memory_saved = st.session_state.initial_memory - current_memory
                if memory_saved > 0:
                    st.sidebar.success(f"Memory optimized! Saved {memory_saved:.1f}MB")
            
        except Exception as e:
            st.warning(f"Memory optimization note: {e}")
    
    def run(self):
        # Header with minimal styling
        st.markdown("""
        <style>
        .main-header {
            font-size: 2.5rem;
            color: #1f77b4;
            text-align: center;
            margin-bottom: 1rem;
        }
        .memory-info {
            font-size: 0.8rem;
            color: #666;
            text-align: center;
        }
        </style>
        """, unsafe_allow_html=True)
        
        st.markdown('<h1 class="main-header">📈 NextTick AI Stock Prediction</h1>', unsafe_allow_html=True)
        
        # Show memory info
        try:
            current_memory = self._get_memory_usage()
            st.markdown(f'<p class="memory-info">Memory usage: {current_memory:.1f}MB</p>', unsafe_allow_html=True)
        except:
            pass
        
        # Sidebar
        st.sidebar.title("Configuration")
        
        # Stock symbol input
        symbol = st.sidebar.text_input("Enter Stock Symbol", "TSLA").upper()
        
        # Analysis period with memory considerations
        period = st.sidebar.selectbox("Data Period", ["1mo", "3mo", "6mo"], index=1)
        
        # Feature selection with memory warnings
        st.sidebar.subheader("Model Features")
        use_technical_indicators = st.sidebar.checkbox("Use Technical Indicators", True)
        use_sentiment_analysis = st.sidebar.checkbox("Use Sentiment Analysis", True)
        use_advanced_model = st.sidebar.checkbox("Use Enhanced Prediction Model", True)
        
        if use_sentiment_analysis:
            st.sidebar.caption("⚠️ Sentiment analysis uses significant memory")
        
        # Memory management
        st.sidebar.subheader("Memory Management")
        
        col1, col2 = st.sidebar.columns(2)
        with col1:
            if st.button("🔄 Clear Cache"):
                self._clear_memory_cache()
                st.rerun()
        
        with col2:
            if st.button("🧹 Force Cleanup"):
                self._force_memory_cleanup()
                st.rerun()
        
        # Display memory info
        if 'analysis_count' in st.session_state:
            st.sidebar.info(f"Analyses since clear: {st.session_state.analysis_count}/3")
        
        # Performance mode
        performance_mode = st.sidebar.checkbox("Performance Mode", False, 
                                             help="Reduces features for better memory usage")
        
        if performance_mode:
            use_technical_indicators = False
            use_sentiment_analysis = False
            st.sidebar.info("Performance mode: Basic features only")
        
        # Main content
        if st.sidebar.button("Analyze Stock") or symbol:
            try:
                self.analyze_stock(symbol, period, use_technical_indicators, use_sentiment_analysis, use_advanced_model, performance_mode)
            except Exception as e:
                self._handle_error(e)
        
        # About section
        st.sidebar.markdown("---")
        st.sidebar.info("""
        **Memory Optimized Version**
        - Proactive memory management
        - Lazy loading of heavy models
        - Automatic cleanup every 3 analyses
        - Memory usage monitoring
        """)
    
    def _handle_error(self, error):
        """Handle errors with memory cleanup"""
        st.error(f"Application error: {str(error)}")
        st.info("💡 Try clearing memory cache or using Performance Mode")
        
        # Force cleanup on error
        self._force_memory_cleanup()
    
    def analyze_stock(self, symbol, period, use_technical_indicators, use_sentiment_analysis, use_advanced_model, performance_mode):
        """Memory-optimized stock analysis"""
        try:
            # Use smart memory management
            self._smart_memory_management()
            
            # Initialize session state for data
            cache_key = f"{symbol}_{period}_{performance_mode}"
            if 'stock_data' not in st.session_state or st.session_state.get('current_cache_key') != cache_key:
                with st.spinner(f"📊 Fetching data for {symbol}..."):
                    # Clear previous data first
                    if 'stock_data' in st.session_state:
                        del st.session_state.stock_data
                    if 'news_data' in st.session_state:
                        del st.session_state.news_data
                    
                    # Fetch new data
                    stock_data = self.data_fetcher.get_stock_data(symbol, period)
                    
                    if stock_data is None or stock_data.empty:
                        st.error(f"❌ No data found for {symbol}")
                        return
                    
                    # Only fetch news if sentiment analysis is enabled
                    news_data = pd.DataFrame()
                    if use_sentiment_analysis and not performance_mode:
                        news_data = self.data_fetcher.get_news_data(symbol)
                    
                    # Store in session state
                    st.session_state.stock_data = stock_data
                    st.session_state.news_data = news_data
                    st.session_state.current_cache_key = cache_key
                    st.session_state.current_symbol = symbol
            
            stock_data = st.session_state.stock_data
            news_data = st.session_state.news_data
            
            # Display basic stock info (lightweight)
            self._display_stock_info(stock_data, symbol)
            
            # Stock price chart
            st.subheader("📊 Stock Price Chart")
            price_chart = self.visualizer.plot_stock_price(stock_data, f"{symbol} Stock Price")
            st.plotly_chart(price_chart, use_container_width=True)
            
            # Technical indicators (optional, memory intensive)
            if use_technical_indicators and not performance_mode:
                st.subheader("🔧 Technical Indicators")
                try:
                    tech_fig = self.visualizer.plot_technical_indicators(stock_data)
                    st.pyplot(tech_fig)
                    plt.close('all')  # Important: close matplotlib figures
                except Exception as e:
                    st.warning(f"Technical indicators unavailable: {e}")
            
            # Sentiment Analysis (memory intensive)
            sentiment_score = 0.0
            if use_sentiment_analysis and not performance_mode:
                st.subheader("😊 Market Sentiment Analysis")
                try:
                    with st.spinner("🤖 Analyzing news sentiment..."):
                        sentiment_score = self.sentiment_analyzer.analyze_news_sentiment(news_data)
                    
                    # Display sentiment gauge
                    sentiment_chart = self.visualizer.plot_sentiment_analysis(sentiment_score, news_data)
                    st.plotly_chart(sentiment_chart, use_container_width=True)
                    
                    # Show sentiment interpretation
                    self._display_sentiment_interpretation(sentiment_score)
                    
                    # Show limited news articles
                    self._display_news_articles(news_data)
                    
                except Exception as e:
                    st.warning(f"Sentiment analysis unavailable: {e}")
            
            # Machine Learning Prediction
            st.subheader("🤖 AI Prediction & Recommendation")
            try:
                # Use basic prediction in performance mode
                if performance_mode:
                    self._generate_basic_prediction(stock_data, sentiment_score, symbol)
                else:
                    if use_advanced_model:
                        self._generate_enhanced_prediction(stock_data, sentiment_score, symbol)
                    else:
                        self._generate_prediction(stock_data, sentiment_score, symbol)
            except Exception as e:
                st.warning(f"Prediction unavailable: {e}")
            
            # Risk Disclaimer
            st.markdown("---")
            st.warning("""
            **Disclaimer:** Educational purposes only. Always conduct your own research.
            """)
            
        except Exception as e:
            self._handle_error(e)
    
    def _display_stock_info(self, stock_data, symbol):
        """Display lightweight stock information"""
        st.subheader(f"📈 {symbol} Stock Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            current_price = stock_data['Close'].iloc[-1]
            st.metric("Current Price", f"${current_price:.2f}")
        
        with col2:
            if len(stock_data) > 1:
                price_change = stock_data['Close'].iloc[-1] - stock_data['Close'].iloc[-2]
                change_percent = (price_change / stock_data['Close'].iloc[-2]) * 100
                st.metric("Daily Change", f"${price_change:.2f}", f"{change_percent:.2f}%")
        
        with col3:
            volume = stock_data['Volume'].iloc[-1]
            st.metric("Volume", f"{volume:,.0f}")
        
        with col4:
            if len(stock_data) > 20 and 'sma_20' in stock_data.columns:
                sma_20 = stock_data['sma_20'].iloc[-1]
                price_ratio = (stock_data['Close'].iloc[-1] / sma_20 - 1) * 100
                st.metric("vs SMA20", f"{price_ratio:+.1f}%")
    
    def _display_sentiment_interpretation(self, sentiment_score):
        """Display lightweight sentiment interpretation"""
        if sentiment_score > 0.3:
            interpretation = "Strongly Bullish"
            color = "green"
        elif sentiment_score > 0.1:
            interpretation = "Bullish" 
            color = "lightgreen"
        elif sentiment_score > -0.1:
            interpretation = "Neutral"
            color = "gray"
        elif sentiment_score > -0.3:
            interpretation = "Bearish"
            color = "orange"
        else:
            interpretation = "Strongly Bearish"
            color = "red"
        
        st.info(f"**Sentiment:** {interpretation} (Score: {sentiment_score:.3f})")
    
    def _display_news_articles(self, news_data):
        """Display limited news articles"""
        st.subheader("📰 Recent News")
        if not news_data.empty:
            # Show only 2 articles to save memory
            for idx, article in news_data.head(2).iterrows():
                with st.expander(f"{article['title'][:80]}..."):
                    st.write(f"**Source:** {article['source']}")
                    st.write(f"**Summary:** {article.get('description', 'No description')[:200]}...")
        else:
            st.info("No recent news available.")
    
    def _generate_enhanced_prediction(self, stock_data, sentiment_score, symbol):
        """Enhanced prediction with memory optimization"""
        with st.spinner("🧠 Training enhanced model..."):
            # Use optimized feature set
            feature_columns = ['Close', 'Volume']
            if 'rsi' in stock_data.columns:
                feature_columns.append('rsi')
            if 'macd' in stock_data.columns:
                feature_columns.append('macd')
            
            # Filter data efficiently
            train_data = stock_data[feature_columns].dropna()
            
            if len(train_data) > 15:
                try:
                    success = self.stock_predictor.train_enhanced_model(train_data, feature_columns)
                    
                    if success:
                        predicted_price, current_price, confidence = self.stock_predictor.predict_next_day(train_data, feature_columns)
                        
                        if predicted_price is not None:
                            self._display_enhanced_prediction_results(
                                predicted_price, current_price, sentiment_score, confidence, symbol
                            )
                            
                            # Show model metrics
                            self._display_model_metrics(train_data, feature_columns)
                    else:
                        st.warning("Enhanced model training failed. Try with more data.")
                        # Fallback to basic prediction
                        self._generate_prediction(stock_data, sentiment_score, symbol)
                        
                except Exception as e:
                    st.warning(f"Enhanced prediction model error: {e}")
                    # Fallback to basic prediction
                    self._generate_prediction(stock_data, sentiment_score, symbol)
            else:
                st.warning("Insufficient data for enhanced prediction. Need at least 15 days of clean data.")
                # Fallback to basic prediction
                self._generate_prediction(stock_data, sentiment_score, symbol)
    
    def _generate_prediction(self, stock_data, sentiment_score, symbol):
        """Basic stock prediction (fallback method)"""
        with st.spinner("🧠 Training model..."):
            # Use simpler features for reliability
            feature_columns = ['Close', 'Volume']
            if 'rsi' in stock_data.columns:
                feature_columns.append('rsi')
            
            # Filter out rows with NaN values
            train_data = stock_data[feature_columns].dropna()
            
            if len(train_data) > 10:
                try:
                    success = self.stock_predictor.train_linear_regression(train_data, feature_columns)
                    
                    if success:
                        predicted_price, current_price = self.stock_predictor.predict_next_day(train_data, feature_columns)
                        
                        if predicted_price is not None:
                            self._display_prediction_results(predicted_price, current_price, sentiment_score)
                    else:
                        st.warning("Model training failed. Try with more data.")
                        
                except Exception as e:
                    st.warning(f"Prediction model error: {e}")
            else:
                st.warning("Insufficient data for accurate prediction.")
    
    def _generate_basic_prediction(self, stock_data, sentiment_score, symbol):
        """Basic prediction for performance mode"""
        with st.spinner("🧠 Calculating prediction..."):
            # Simple price-based prediction
            current_price = stock_data['Close'].iloc[-1]
            
            # Simple moving average prediction
            if len(stock_data) > 5:
                short_ma = stock_data['Close'].tail(5).mean()
                trend = "up" if current_price > short_ma else "down"
                
                # Simple prediction based on recent trend
                if trend == "up":
                    predicted_price = current_price * 1.01  # +1%
                else:
                    predicted_price = current_price * 0.99  # -1%
                
                self._display_basic_prediction_results(predicted_price, current_price, sentiment_score, trend)
            else:
                st.warning("Insufficient data for basic prediction.")
    
    def _display_enhanced_prediction_results(self, predicted_price, current_price, sentiment_score, confidence, symbol):
        """Fixed recommendation system with accurate buy/sell logic"""
        price_change_pred = ((predicted_price - current_price) / current_price) * 100
        
        # Improved recommendation algorithm with proper risk assessment
        price_trend_strength = abs(price_change_pred)
        sentiment_strength = abs(sentiment_score)
        
        # Define clear thresholds
        STRONG_PRICE_THRESHOLD = 2.0    # 2% price movement is significant
        MODERATE_PRICE_THRESHOLD = 1.0  # 1% price movement is moderate
        STRONG_SENTIMENT_THRESHOLD = 0.3
        MODERATE_SENTIMENT_THRESHOLD = 0.15
        
        # Calculate signal strength and direction
        price_signal = 1 if price_change_pred > 0 else -1
        sentiment_signal = 1 if sentiment_score > 0 else -1
        
        # Weight factors (price prediction is more important than sentiment)
        price_weight = 0.7
        sentiment_weight = 0.3
        
        # Calculate weighted score
        price_component = price_change_pred * price_weight
        sentiment_component = sentiment_score * 100 * sentiment_weight
        combined_score = price_component + sentiment_component
        
        # CONFIDENCE-BASED RECOMMENDATION LOGIC
        
        # Case 1: Strong agreement between price and sentiment
        if price_signal == sentiment_signal:
            if price_trend_strength > STRONG_PRICE_THRESHOLD and sentiment_strength > STRONG_SENTIMENT_THRESHOLD:
                # Strong bullish or bearish consensus
                if price_signal > 0:
                    recommendation = "STRONG BUY"
                    recommendation_color = "green"
                    recommendation_emoji = "🚀"
                    reasoning = "Strong price growth expected with positive market sentiment"
                else:
                    recommendation = "STRONG SELL"
                    recommendation_color = "red"
                    recommendation_emoji = "🔻"
                    reasoning = "Significant price decline expected with negative market sentiment"
                    
            elif price_trend_strength > MODERATE_PRICE_THRESHOLD and sentiment_strength > MODERATE_SENTIMENT_THRESHOLD:
                # Moderate consensus
                if price_signal > 0:
                    recommendation = "BUY"
                    recommendation_color = "lightgreen"
                    recommendation_emoji = "📈"
                    reasoning = "Moderate price growth expected with favorable sentiment"
                else:
                    recommendation = "SELL"
                    recommendation_color = "orange"
                    recommendation_emoji = "📉"
                    reasoning = "Price decline expected with concerning market sentiment"
                    
            else:
                # Weak consensus - be cautious
                recommendation = "HOLD"
                recommendation_color = "gray"
                recommendation_emoji = "⚖️"
                reasoning = "Weak signals in both price prediction and market sentiment"
        
        # Case 2: Mixed signals (price and sentiment disagree)
        else:
            # When signals conflict, price prediction gets priority but we're more cautious
            if price_trend_strength > STRONG_PRICE_THRESHOLD:
                # Strong price trend overrides sentiment
                if price_signal > 0:
                    recommendation = "CAUTIOUS BUY"
                    recommendation_color = "lightgreen"
                    recommendation_emoji = "📈"
                    reasoning = "Strong price growth expected despite mixed sentiment"
                else:
                    recommendation = "CAUTIOUS SELL"
                    recommendation_color = "orange"
                    recommendation_emoji = "📉"
                    reasoning = "Significant price decline expected despite mixed sentiment"
                    
            elif sentiment_strength > STRONG_SENTIMENT_THRESHOLD:
                # Very strong sentiment might override weak price signal
                if sentiment_signal > 0:
                    recommendation = "SENTIMENT-BIASED HOLD"
                    recommendation_color = "lightblue"
                    recommendation_emoji = "😊"
                    reasoning = "Strong positive sentiment but weak price signal - monitor closely"
                else:
                    recommendation = "SENTIMENT-BIASED HOLD"
                    recommendation_color = "lightcoral"
                    recommendation_emoji = "😞"
                    reasoning = "Strong negative sentiment but weak price signal - monitor closely"
                    
            else:
                # Weak conflicting signals - definitely hold
                recommendation = "HOLD"
                recommendation_color = "gray"
                recommendation_emoji = "⚖️"
                reasoning = "Conflicting weak signals - wait for clearer direction"
        
        # ADJUST FOR CONFIDENCE LEVEL
        if confidence and confidence < 0.6:
            # Low confidence overrides to HOLD regardless of other signals
            if recommendation not in ["HOLD", "SENTIMENT-BIASED HOLD"]:
                recommendation = f"HOLD (LOW CONFIDENCE: {confidence*100:.1f}%)"
                recommendation_color = "gray"
                recommendation_emoji = "🤔"
                reasoning += f" - Model confidence is low ({confidence*100:.1f}%)"
        
        # SPECIAL CASES that always result in HOLD
        if (abs(price_change_pred) < 0.5 and abs(sentiment_score) < 0.1) or \
           (abs(combined_score) < 1.0 and confidence and confidence < 0.7):
            recommendation = "HOLD"
            recommendation_color = "gray"
            recommendation_emoji = "⚖️"
            reasoning = "Insufficient signal strength for clear recommendation"
        
        # Display results
        st.subheader("🎯 Prediction Results")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Current Price", f"${current_price:.2f}")
        
        with col2:
            st.metric("Predicted Price", f"${predicted_price:.2f}", f"{price_change_pred:+.2f}%")
        
        with col3:
            # Color code sentiment
            sentiment_display = f"{sentiment_score:.3f}"
            st.metric("Sentiment Score", sentiment_display)
        
        with col4:
            confidence_display = confidence * 100 if confidence else "N/A"
            if confidence:
                st.metric("Model Confidence", f"{confidence_display:.1f}%")
            else:
                st.metric("Model Confidence", "N/A")
        
        # Display recommendation with detailed reasoning
        st.markdown(f"""
        <div style='background-color: #f8f9fa; padding: 1.5rem; border-radius: 10px; border-left: 5px solid {recommendation_color}; margin: 1rem 0;'>
            <h3 style='color: {recommendation_color}; margin: 0;'>{recommendation_emoji} Recommendation: {recommendation}</h3>
            <p style='margin: 0.5rem 0 0 0; color: #666;'><strong>Reasoning:</strong> {reasoning}</p>
            <p style='margin: 0.5rem 0 0 0; font-size: 0.9rem; color: #888;'>
                Price Change: {price_change_pred:+.2f}% | Sentiment: {sentiment_score:.3f} | Combined Score: {combined_score:.2f}
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Add risk assessment
        self._display_risk_assessment(price_change_pred, sentiment_score, confidence, recommendation)
    
    def _display_risk_assessment(self, price_change_pred, sentiment_score, confidence, recommendation):
        """Display risk assessment based on the signals"""
        
        risk_level = "MEDIUM"
        risk_color = "orange"
        risk_details = []
        
        # Calculate risk factors
        if abs(price_change_pred) > 3.0:
            risk_details.append("High price volatility expected")
            risk_level = "HIGH"
            risk_color = "red"
        elif abs(price_change_pred) < 0.5:
            risk_details.append("Low price movement expected")
            risk_level = "LOW"
            risk_color = "green"
        
        if abs(sentiment_score) > 0.3:
            risk_details.append("Strong market sentiment detected")
            if risk_level != "HIGH":
                risk_level = "MEDIUM_HIGH"
        elif abs(sentiment_score) < 0.1:
            risk_details.append("Neutral market sentiment")
            if risk_level == "MEDIUM":
                risk_level = "LOW_MEDIUM"
        
        if confidence and confidence < 0.6:
            risk_details.append("Low model confidence")
            risk_level = "HIGH" if risk_level != "LOW" else "MEDIUM"
            risk_color = "red"
        
        # Special case for HOLD recommendations
        if recommendation.startswith("HOLD"):
            risk_level = "LOW"
            risk_color = "green"
            risk_details.append("Conservative recommendation reduces risk")
        
        risk_details_text = " • ".join(risk_details) if risk_details else "Standard market conditions"
        
        st.markdown(f"""
        <div style='background-color: #fff3cd; padding: 1rem; border-radius: 8px; border-left: 4px solid {risk_color}; margin: 1rem 0;'>
            <h4 style='color: {risk_color}; margin: 0;'>📊 Risk Assessment: {risk_level}</h4>
            <p style='margin: 0.5rem 0 0 0; color: #856404;'>{risk_details_text}</p>
        </div>
        """, unsafe_allow_html=True)
    
    def _display_prediction_results(self, predicted_price, current_price, sentiment_score):
        """Display basic prediction results with improved logic"""
        price_change_pred = ((predicted_price - current_price) / current_price) * 100
        
        # Improved basic recommendation logic
        if price_change_pred > 2.0 and sentiment_score > 0.2:
            recommendation = "BUY"
            recommendation_color = "green"
            reasoning = "Strong price growth with positive sentiment"
        elif price_change_pred > 1.0 and sentiment_score > 0.1:
            recommendation = "CAUTIOUS BUY"
            recommendation_color = "lightgreen"
            reasoning = "Moderate price growth with favorable sentiment"
        elif price_change_pred < -2.0 and sentiment_score < -0.2:
            recommendation = "SELL"
            recommendation_color = "red"
            reasoning = "Significant decline with negative sentiment"
        elif price_change_pred < -1.0 and sentiment_score < -0.1:
            recommendation = "CAUTIOUS SELL"
            recommendation_color = "orange"
            reasoning = "Price decline with concerning sentiment"
        elif price_change_pred > 1.5:
            recommendation = "PRICE-BIASED HOLD"
            recommendation_color = "lightblue"
            reasoning = "Price growth expected but sentiment is neutral"
        elif price_change_pred < -1.5:
            recommendation = "PRICE-BIASED HOLD"
            recommendation_color = "lightcoral"
            reasoning = "Price decline expected but sentiment is neutral"
        elif sentiment_score > 0.25:
            recommendation = "SENTIMENT-BIASED HOLD"
            recommendation_color = "lightblue"
            reasoning = "Positive sentiment but weak price signal"
        elif sentiment_score < -0.25:
            recommendation = "SENTIMENT-BIASED HOLD"
            recommendation_color = "lightcoral"
            reasoning = "Negative sentiment but weak price signal"
        else:
            recommendation = "HOLD"
            recommendation_color = "gray"
            reasoning = "Insufficient signals for clear direction"
        
        # Display results
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Current Price", f"${current_price:.2f}")
        
        with col2:
            st.metric("Predicted Price", f"${predicted_price:.2f}", f"{price_change_pred:+.2f}%")
        
        with col3:
            st.metric("Sentiment Score", f"{sentiment_score:.3f}")
        
        # Enhanced recommendation display
        st.markdown(f"""
        <div style='background-color: #f0f2f6; padding: 1.5rem; border-radius: 10px; border-left: 5px solid {recommendation_color}; margin: 1rem 0;'>
            <h3 style='color: {recommendation_color}; margin: 0;'>Recommendation: {recommendation}</h3>
            <p style='margin: 0.5rem 0 0 0;'><strong>Reasoning:</strong> {reasoning}</p>
        </div>
        """, unsafe_allow_html=True)
    
    def _display_basic_prediction_results(self, predicted_price, current_price, sentiment_score, trend):
        """Display basic prediction results for performance mode"""
        price_change_pred = ((predicted_price - current_price) / current_price) * 100
        
        # Simple recommendation based on trend
        if trend == "up" and sentiment_score > 0:
            recommendation = "BUY"
            color = "green"
        elif trend == "down" and sentiment_score < 0:
            recommendation = "SELL"
            color = "red"
        else:
            recommendation = "HOLD"
            color = "gray"
        
        # Display results
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Current Price", f"${current_price:.2f}")
        
        with col2:
            st.metric("Predicted Price", f"${predicted_price:.2f}", f"{price_change_pred:+.2f}%")
        
        st.info(f"**Recommendation:** {recommendation} (Based on {trend} trend)")
    
    def _display_model_metrics(self, train_data, feature_columns):
        """Display model performance metrics"""
        try:
            metrics = self.stock_predictor.calculate_enhanced_metrics(train_data, feature_columns)
            if metrics:
                with st.expander("📊 Model Performance Metrics"):
                    st.info("These metrics show how well the model performs on historical data:")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        r2 = metrics.get('R2_Score', 0)
                        st.metric("R² Score", f"{r2:.3f}", 
                                 delta="Good" if r2 > 0.7 else "Fair" if r2 > 0.4 else "Poor",
                                 delta_color="normal")
                    
                    with col2:
                        rmse = metrics.get('RMSE', 0)
                        st.metric("RMSE", f"{rmse:.3f}")
                    
                    with col3:
                        mae = metrics.get('MAE', 0)
                        st.metric("MAE", f"{mae:.3f}")
                    
                    with col4:
                        accuracy = metrics.get('Accuracy_Score', 0) * 100
                        st.metric("Accuracy", f"{accuracy:.1f}%")
        except Exception as e:
            # Silently fail if metrics can't be calculated
            pass

# Run the application
if __name__ == "__main__":
    app = MemoryOptimizedNextTickApp()
    app.run()
