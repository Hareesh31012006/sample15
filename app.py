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
    
    def run(self):
        # Header
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
        
        # Memory management
        if st.sidebar.button("🔄 Clear Memory Cache"):
            self._clear_memory_cache()
            st.rerun()
        
        # Main content
        if st.sidebar.button("Analyze Stock") or symbol:
            try:
                self.analyze_stock(symbol, period, use_technical_indicators, use_sentiment_analysis)
            except Exception as e:
                self._handle_error(e)
        
        # About section
        st.sidebar.markdown("---")
        st.sidebar.info("""
        **Memory Optimized Version**
        - Lazy loading of ML models
        - Quantized sentiment analysis
        - Efficient memory management
        """)
    
    def _clear_memory_cache(self):
        """Clear memory cache efficiently"""
        import gc
        if self.sentiment_analyzer:
            self.sentiment_analyzer.cleanup()
        
        # Clear session state selectively
        keys_to_keep = ['current_symbol', 'current_cache_key']
        keys_to_delete = [key for key in st.session_state.keys() if key not in keys_to_keep]
        
        for key in keys_to_delete:
            del st.session_state[key]
        
        gc.collect()
        st.success("Memory cache cleared!")
    
    def _handle_error(self, error):
        """Handle errors with memory cleanup"""
        st.error(f"Application error: {str(error)}")
        st.info("💡 Try clearing the memory cache or using a shorter data period.")
        
        # Force cleanup on error
        self._clear_memory_cache()
    
    def analyze_stock(self, symbol, period, use_technical_indicators, use_sentiment_analysis):
        """Optimized stock analysis with memory management"""
        try:
            # Initialize session state for data
            cache_key = f"{symbol}_{period}"
            if 'stock_data' not in st.session_state or st.session_state.get('current_cache_key') != cache_key:
                with st.spinner(f"📊 Fetching data for {symbol}..."):
                    # Fetch stock data
                    stock_data = self.data_fetcher.get_stock_data(symbol, period)
                    
                    if stock_data is None or stock_data.empty:
                        st.error(f"❌ No data found for {symbol}")
                        return
                    
                    # Fetch news data
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
            st.subheader("Stock Price Chart")
            price_chart = self.visualizer.plot_stock_price(stock_data, f"{symbol} Stock Price")
            st.plotly_chart(price_chart, use_container_width=True)
            
            # Technical indicators
            if use_technical_indicators:
                st.subheader("Technical Indicators")
                try:
                    tech_fig = self.visualizer.plot_technical_indicators(stock_data)
                    st.pyplot(tech_fig)
                except Exception as e:
                    st.warning(f"Technical indicators unavailable: {e}")
            
            # Sentiment Analysis (with memory optimization)
            sentiment_score = 0.0
            if use_sentiment_analysis:
                st.subheader("Market Sentiment Analysis")
                try:
                    with st.spinner("🤖 Analyzing news sentiment..."):
                        sentiment_score = self.sentiment_analyzer.analyze_news_sentiment(news_data)
                    
                    # Display sentiment gauge
                    sentiment_chart = self.visualizer.plot_sentiment_analysis(sentiment_score, news_data)
                    st.plotly_chart(sentiment_chart, use_container_width=True)
                    
                    # Show news articles
                    self._display_news_articles(news_data)
                    
                except Exception as e:
                    st.warning(f"Sentiment analysis unavailable: {e}")
            
            # Machine Learning Prediction
            st.subheader("AI Prediction & Recommendation")
            try:
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
        """Display basic stock information"""
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
            st.metric("Volume", f"{volume:,}")
        
        with col4:
            if 'rsi' in stock_data.columns:
                rsi = stock_data['rsi'].iloc[-1]
                st.metric("RSI", f"{rsi:.2f}")
    
    def _display_news_articles(self, news_data):
        """Display news articles"""
        st.subheader("Recent News Articles")
        if not news_data.empty:
            for idx, article in news_data.head(3).iterrows():  # Show only 3 articles
                with st.expander(f"{article['title']} - {article['source']}"):
                    st.write(f"**Published:** {article['published_at']}")
                    st.write(f"**Description:** {article.get('description', 'No description')}")
        else:
            st.info("No recent news articles available.")
    
    def _generate_prediction(self, stock_data, sentiment_score, symbol):
        """Generate stock prediction with memory optimization"""
        with st.spinner("🧠 Training model and making prediction..."):
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
    
    def _display_prediction_results(self, predicted_price, current_price, sentiment_score):
        """Display prediction results"""
        price_change_pred = ((predicted_price - current_price) / current_price) * 100
        
        # Determine recommendation
        if price_change_pred > 1.0 and sentiment_score > 0.1:
            recommendation = "BUY"
            recommendation_color = "green"
        elif price_change_pred < -1.0 or sentiment_score < -0.1:
            recommendation = "SELL"
            recommendation_color = "red"
        else:
            recommendation = "HOLD"
            recommendation_color = "orange"
        
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
        <div style='background-color: #f0f2f6; padding: 1.5rem; border-radius: 10px; border-left: 5px solid {recommendation_color}; margin: 1rem 0;'>
            <h3 style='color: {recommendation_color}; margin: 0;'>Recommendation: {recommendation}</h3>
            <p style='margin: 0.5rem 0 0 0;'>Based on AI prediction and market sentiment</p>
        </div>
        """, unsafe_allow_html=True)

# Run the application
if __name__ == "__main__":
    app = MemoryOptimizedNextTickApp()
    app.run()
