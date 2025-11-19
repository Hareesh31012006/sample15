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
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024  # MB
    
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
            st.sidebar.info("Performance mode: Basic features only")
        
        # Main content
        if st.sidebar.button("Analyze Stock") or symbol:
            try:
                self.analyze_stock(symbol, period, use_technical_indicators, use_sentiment_analysis, performance_mode)
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
    
    def analyze_stock(self, symbol, period, use_technical_indicators, use_sentiment_analysis, performance_mode):
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
                    if use_sentiment_analysis:
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
                    self._generate_enhanced_prediction(stock_data, sentiment_score, symbol)
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
        with st.spinner("🧠 Training model..."):
            # Use optimized feature set
            feature_columns = ['Close', 'Volume']
            if 'rsi' in stock_data.columns and not performance_mode:
                feature_columns.append('rsi')
            
            # Filter data efficiently
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
                st.warning("Insufficient data for prediction.")
    
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
    
    def _display_prediction_results(self, predicted_price, current_price, sentiment_score):
        """Display prediction results"""
        price_change_pred = ((predicted_price - current_price) / current_price) * 100
        
        # Smart recommendation logic
        if price_change_pred > 2.0 and sentiment_score > 0.2:
            recommendation = "STRONG BUY"
            color = "green"
        elif price_change_pred > 0.5 and sentiment_score > 0.05:
            recommendation = "BUY"
            color = "lightgreen"
        elif price_change_pred < -2.0 and sentiment_score < -0.2:
            recommendation = "STRONG SELL"
            color = "red"
        elif price_change_pred < -0.5 and sentiment_score < -0.05:
            recommendation = "SELL"
            color = "orange"
        elif price_change_pred > 1.0:
            recommendation = "CAUTIOUS BUY"
            color = "lightgreen"
        elif price_change_pred < -1.0:
            recommendation = "CAUTIOUS SELL"
            color = "orange"
        else:
            recommendation = "HOLD"
            color = "gray"
        
        # Display results
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Current Price", f"${current_price:.2f}")
        
        with col2:
            st.metric("Predicted Price", f"${predicted_price:.2f}", f"{price_change_pred:+.2f}%")
        
        with col3:
            st.metric("Sentiment", f"{sentiment_score:.3f}")
        
        # Recommendation
        st.markdown(f"""
        <div style='padding: 1rem; border-radius: 10px; border-left: 5px solid {color}; margin: 1rem 0;'>
            <h3 style='color: {color}; margin: 0;'>Recommendation: {recommendation}</h3>
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

# Run the application
if __name__ == "__main__":
    app = MemoryOptimizedNextTickApp()
    app.run()
