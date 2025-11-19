import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
import gc
import warnings
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
            self.data_fetcher = DataFetcher()
            self.sentiment_analyzer = OptimizedSentimentAnalyzer()
            self.stock_predictor = StockPredictor()
            self.visualizer = StockVisualizer()
        except Exception as e:
            st.error(f"Error initializing app: {e}")
    
    def _get_memory_usage(self):
        """Get current memory usage"""
        try:
            import psutil
            process = psutil.Process(os.getpid())
            return process.memory_info().rss / 1024 / 1024  # MB
        except:
            return 0
    
    def _check_memory_threshold(self):
        """Check if memory usage is too high"""
        try:
            current_memory = self._get_memory_usage()
            if current_memory > 500:  # 500MB threshold
                st.warning(f"🔄 High memory usage detected ({current_memory:.1f}MB). Clearing cache...")
                self._force_memory_cleanup()
                return True
            return False
        except:
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
            
            # Force garbage collection
            gc.collect()
            
            st.success("🧹 Memory cleanup completed!")
            
        except Exception as e:
            st.warning(f"Memory cleanup warning: {e}")
    
    def _smart_memory_management(self):
        """Smart memory management with proactive cleanup"""
        if 'analysis_count' not in st.session_state:
            st.session_state.analysis_count = 0
            st.session_state.last_memory_clear = datetime.now()
        
        st.session_state.analysis_count += 1
        
        # Check memory threshold first
        if self._check_memory_threshold():
            return
        
        # Clear memory every 3 analyses or 3 minutes
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
                if key not in essential_keys:
                    del st.session_state[key]
            
            # Force garbage collection
            gc.collect()
            
            st.success("🔄 Memory optimized!")
            
        except Exception as e:
            st.warning(f"Memory optimization note: {e}")
    
    def run(self):
        # Header
        st.markdown("""
        <style>
        .main-header {
            font-size: 2.5rem;
            color: #1f77b4;
            text-align: center;
            margin-bottom: 1rem;
        }
        </style>
        """, unsafe_allow_html=True)
        
        st.markdown('<h1 class="main-header">📈 NextTick AI Stock Prediction</h1>', unsafe_allow_html=True)
        
        # Sidebar
        st.sidebar.title("Configuration")
        
        # Stock symbol input
        symbol = st.sidebar.text_input("Enter Stock Symbol", "TSLA").upper()
        
        # Analysis period
        period = st.sidebar.selectbox("Data Period", ["1mo", "3mo", "6mo"], index=1)
        
        # Feature selection
        st.sidebar.subheader("Model Features")
        use_technical_indicators = st.sidebar.checkbox("Use Technical Indicators", True)
        use_sentiment_analysis = st.sidebar.checkbox("Use Sentiment Analysis", True)
        use_advanced_model = st.sidebar.checkbox("Use Enhanced Prediction Model", True)
        
        # Memory management
        st.sidebar.subheader("Memory Management")
        
        if st.sidebar.button("🔄 Clear Memory Cache"):
            self._clear_memory_cache()
            st.rerun()
        
        # Display memory info
        if 'analysis_count' in st.session_state:
            st.sidebar.info(f"Analyses since clear: {st.session_state.analysis_count}/3")
        
        # Main content
        if st.sidebar.button("Analyze Stock") or symbol:
            try:
                self.analyze_stock(symbol, period, use_technical_indicators, use_sentiment_analysis, use_advanced_model)
            except Exception as e:
                self._handle_error(e)
        
        # About section
        st.sidebar.markdown("---")
        st.sidebar.info("""
        **Stock Prediction App**
        - AI-powered stock analysis
        - Technical indicators
        - Sentiment analysis
        - Memory optimized
        """)
    
    def _handle_error(self, error):
        """Handle errors with memory cleanup"""
        st.error(f"Application error: {str(error)}")
        st.info("💡 Try clearing the memory cache or using a shorter data period.")
        
        # Force cleanup on error
        self._force_memory_cleanup()
    
    def analyze_stock(self, symbol, period, use_technical_indicators, use_sentiment_analysis, use_advanced_model):
        """Optimized stock analysis with smart memory management"""
        try:
            # Use smart memory management
            self._smart_memory_management()
            
            # Initialize session state for data
            cache_key = f"{symbol}_{period}"
            if 'stock_data' not in st.session_state or st.session_state.get('current_cache_key') != cache_key:
                with st.spinner(f"📊 Fetching data for {symbol}..."):
                    # Fetch new data
                    stock_data = self.data_fetcher.get_stock_data(symbol, period)
                    
                    if stock_data is None or stock_data.empty:
                        st.error(f"❌ No data found for {symbol}")
                        return
                    
                    # Only fetch news if sentiment analysis is enabled
                    news_data = pd.DataFrame()
                    if use_sentiment_analysis:
                        news_data = self.data_fetcher.get_news_data(symbol)
                    
                    # Store in session state
                    st.session_state.stock_data = stock_data
                    st.session_state.news_data = news_data
                    st.session_state.current_cache_key = cache_key
                    st.session_state.current_symbol = symbol
            
            stock_data = st.session_state.stock_data
            news_data = st.session_state.news_data
            
            # Display basic stock info
            self._display_stock_info(stock_data, symbol)
            
            # Stock price chart
            st.subheader("📊 Stock Price Chart")
            price_chart = self.visualizer.plot_stock_price(stock_data, f"{symbol} Stock Price")
            st.plotly_chart(price_chart, use_container_width=True)
            
            # Technical indicators
            if use_technical_indicators:
                st.subheader("🔧 Technical Indicators")
                try:
                    tech_fig = self.visualizer.plot_technical_indicators(stock_data)
                    st.pyplot(tech_fig)
                except Exception as e:
                    st.warning(f"Technical indicators unavailable: {e}")
            
            # Sentiment Analysis
            sentiment_score = 0.0
            if use_sentiment_analysis:
                st.subheader("😊 Market Sentiment Analysis")
                try:
                    with st.spinner("🤖 Analyzing news sentiment..."):
                        sentiment_score = self.sentiment_analyzer.analyze_news_sentiment(news_data)
                    
                    # Display sentiment gauge
                    sentiment_chart = self.visualizer.plot_sentiment_analysis(sentiment_score, news_data)
                    st.plotly_chart(sentiment_chart, use_container_width=True)
                    
                    # Show sentiment interpretation
                    self._display_sentiment_interpretation(sentiment_score)
                    
                    # Show news articles
                    self._display_news_articles(news_data)
                    
                except Exception as e:
                    st.warning(f"Sentiment analysis unavailable: {e}")
            
            # Machine Learning Prediction
            st.subheader("🤖 AI Prediction & Recommendation")
            try:
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
        """Display stock information"""
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
        """Display sentiment interpretation"""
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
        """Display news articles"""
        st.subheader("📰 Recent News")
        if not news_data.empty:
            for idx, article in news_data.head(3).iterrows():
                with st.expander(f"{article['title'][:80]}..."):
                    st.write(f"**Source:** {article['source']}")
                    st.write(f"**Summary:** {article.get('description', 'No description')[:200]}...")
        else:
            st.info("No recent news available.")
    
    def _generate_enhanced_prediction(self, stock_data, sentiment_score, symbol):
        """Enhanced prediction with memory optimization"""
        with st.spinner("🧠 Training enhanced model..."):
            # Use enhanced feature set
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
                    else:
                        st.warning("Enhanced model training failed. Using basic model...")
                        self._generate_prediction(stock_data, sentiment_score, symbol)
                        
                except Exception as e:
                    st.warning(f"Enhanced prediction error: {e}")
                    self._generate_prediction(stock_data, sentiment_score, symbol)
            else:
                st.warning("Insufficient data for enhanced prediction. Using basic model...")
                self._generate_prediction(stock_data, sentiment_score, symbol)
    
    def _generate_prediction(self, stock_data, sentiment_score, symbol):
        """Basic stock prediction"""
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
    
    def _display_enhanced_prediction_results(self, predicted_price, current_price, sentiment_score, confidence, symbol):
        """Fixed recommendation system with accurate buy/sell logic"""
        price_change_pred = ((predicted_price - current_price) / current_price) * 100
        
        # Clear thresholds for decision making
        STRONG_PRICE_THRESHOLD = 2.0
        MODERATE_PRICE_THRESHOLD = 1.0
        STRONG_SENTIMENT_THRESHOLD = 0.3
        MODERATE_SENTIMENT_THRESHOLD = 0.15
        
        price_trend_strength = abs(price_change_pred)
        sentiment_strength = abs(sentiment_score)
        
        # Signal directions
        price_signal = 1 if price_change_pred > 0 else -1
        sentiment_signal = 1 if sentiment_score > 0 else -1
        
        # CONFIDENCE-BASED RECOMMENDATION LOGIC
        recommendation = "HOLD"
        recommendation_color = "gray"
        recommendation_emoji = "⚖️"
        reasoning = "Insufficient signal strength for clear recommendation"
        
        # Case 1: Strong agreement between price and sentiment
        if price_signal == sentiment_signal:
            if price_trend_strength > STRONG_PRICE_THRESHOLD and sentiment_strength > STRONG_SENTIMENT_THRESHOLD:
                if price_signal > 0:
                    recommendation = "STRONG BUY"
                    recommendation_color = "green"
                    recommendation_emoji = "🚀"
                    reasoning = "Strong price growth with very positive sentiment"
                else:
                    recommendation = "STRONG SELL"
                    recommendation_color = "red"
                    recommendation_emoji = "🔻"
                    reasoning = "Significant price decline with very negative sentiment"
                    
            elif price_trend_strength > MODERATE_PRICE_THRESHOLD and sentiment_strength > MODERATE_SENTIMENT_THRESHOLD:
                if price_signal > 0:
                    recommendation = "BUY"
                    recommendation_color = "lightgreen"
                    recommendation_emoji = "📈"
                    reasoning = "Moderate price growth with positive sentiment"
                else:
                    recommendation = "SELL"
                    recommendation_color = "orange"
                    recommendation_emoji = "📉"
                    reasoning = "Price decline with negative sentiment"
        
        # Case 2: Mixed signals
        else:
            if price_trend_strength > STRONG_PRICE_THRESHOLD:
                # Strong price trend gets priority
                if price_signal > 0:
                    recommendation = "CAUTIOUS BUY"
                    recommendation_color = "lightgreen"
                    recommendation_emoji = "📈"
                    reasoning = "Strong price growth despite mixed sentiment"
                else:
                    recommendation = "CAUTIOUS SELL"
                    recommendation_color = "orange"
                    recommendation_emoji = "📉"
                    reasoning = "Significant price decline despite mixed sentiment"
                    
            elif sentiment_strength > STRONG_SENTIMENT_THRESHOLD:
                # Very strong sentiment might influence
                if sentiment_signal > 0:
                    recommendation = "SENTIMENT HOLD"
                    recommendation_color = "lightblue"
                    recommendation_emoji = "😊"
                    reasoning = "Very positive sentiment but weak price signal"
                else:
                    recommendation = "SENTIMENT HOLD"
                    recommendation_color = "lightcoral"
                    recommendation_emoji = "😞"
                    reasoning = "Very negative sentiment but weak price signal"
        
        # Adjust for confidence level
        if confidence and confidence < 0.6:
            if recommendation not in ["HOLD", "SENTIMENT HOLD"]:
                recommendation = f"HOLD (Low Confidence)"
                recommendation_color = "gray"
                recommendation_emoji = "🤔"
                reasoning = f"Model confidence too low ({confidence*100:.1f}%) for trading recommendation"
        
        # Special cases that always result in HOLD
        if (abs(price_change_pred) < 0.5 and abs(sentiment_score) < 0.1):
            recommendation = "HOLD"
            recommendation_color = "gray"
            recommendation_emoji = "⚖️"
            reasoning = "Very weak signals - wait for clearer direction"
        
        # Display results
        st.subheader("🎯 Prediction Results")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Current Price", f"${current_price:.2f}")
        
        with col2:
            st.metric("Predicted Price", f"${predicted_price:.2f}", f"{price_change_pred:+.2f}%")
        
        with col3:
            st.metric("Sentiment Score", f"{sentiment_score:.3f}")
        
        with col4:
            confidence_display = confidence * 100 if confidence else "N/A"
            if confidence:
                st.metric("Model Confidence", f"{confidence_display:.1f}%")
            else:
                st.metric("Model Confidence", "N/A")
        
        # Display recommendation
        st.markdown(f"""
        <div style='background-color: #f8f9fa; padding: 1.5rem; border-radius: 10px; border-left: 5px solid {recommendation_color}; margin: 1rem 0;'>
            <h3 style='color: {recommendation_color}; margin: 0;'>{recommendation_emoji} Recommendation: {recommendation}</h3>
            <p style='margin: 0.5rem 0 0 0; color: #666;'><strong>Reasoning:</strong> {reasoning}</p>
        </div>
        """, unsafe_allow_html=True)
    
    def _display_prediction_results(self, predicted_price, current_price, sentiment_score):
        """Display basic prediction results"""
        price_change_pred = ((predicted_price - current_price) / current_price) * 100
        
        # Conservative recommendation logic
        if price_change_pred > 2.0 and sentiment_score > 0.2:
            recommendation = "BUY"
            color = "green"
            reasoning = "Strong price growth with positive sentiment"
        elif price_change_pred > 1.0 and sentiment_score > 0.1:
            recommendation = "CAUTIOUS BUY"
            color = "lightgreen"
            reasoning = "Moderate price growth with favorable sentiment"
        elif price_change_pred < -2.0 and sentiment_score < -0.2:
            recommendation = "SELL"
            color = "red"
            reasoning = "Significant decline with negative sentiment"
        elif price_change_pred < -1.0 and sentiment_score < -0.1:
            recommendation = "CAUTIOUS SELL"
            color = "orange"
            reasoning = "Price decline with concerning sentiment"
        else:
            recommendation = "HOLD"
            color = "gray"
            reasoning = "Insufficient signals for clear direction"
        
        # Display results
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Current Price", f"${current_price:.2f}")
        
        with col2:
            st.metric("Predicted Price", f"${predicted_price:.2f}", f"{price_change_pred:+.2f}%")
        
        with col3:
            st.metric("Sentiment Score", f"{sentiment_score:.3f}")
        
        # Recommendation
        st.markdown(f"""
        <div style='background-color: #f0f2f6; padding: 1.5rem; border-radius: 10px; border-left: 5px solid {color}; margin: 1rem 0;'>
            <h3 style='color: {color}; margin: 0;'>Recommendation: {recommendation}</h3>
            <p style='margin: 0.5rem 0 0 0;'><strong>Reasoning:</strong> {reasoning}</p>
        </div>
        """, unsafe_allow_html=True)

# Run the application
if __name__ == "__main__":
    app = MemoryOptimizedNextTickApp()
    app.run()
