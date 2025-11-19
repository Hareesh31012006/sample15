import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv

# Import custom modules
from utils.data_fetcher import DataFetcher
from utils.sentiment_analyzer import SentimentAnalyzer
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

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .buy-recommendation {
        background-color: #d4edda;
        border-left: 5px solid #28a745;
    }
    .sell-recommendation {
        background-color: #f8d7da;
        border-left: 5px solid #dc3545;
    }
    .hold-recommendation {
        background-color: #fff3cd;
        border-left: 5px solid #ffc107;
    }
</style>
""", unsafe_allow_html=True)

class NextTickApp:
    def __init__(self):
        self.data_fetcher = DataFetcher()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.stock_predictor = StockPredictor()
        self.visualizer = StockVisualizer()
        
    def run(self):
        # Header
        st.markdown('<h1 class="main-header">📈 NextTick AI Stock Prediction</h1>', unsafe_allow_html=True)
        
        # Sidebar
        st.sidebar.title("Configuration")
        
        # Stock symbol input
        symbol = st.sidebar.text_input("Enter Stock Symbol", "AAPL").upper()
        
        # Analysis period
        period = st.sidebar.selectbox(
            "Data Period",
            ["1mo", "3mo", "6mo", "1y", "2y"],
            index=2
        )
        
        # Feature selection
        st.sidebar.subheader("Model Features")
        use_technical_indicators = st.sidebar.checkbox("Use Technical Indicators", True)
        use_sentiment_analysis = st.sidebar.checkbox("Use Sentiment Analysis", True)
        
        # Main content
        if st.sidebar.button("Analyze Stock") or symbol:
            self.analyze_stock(symbol, period, use_technical_indicators, use_sentiment_analysis)
        
        # About section in sidebar
        st.sidebar.markdown("---")
        st.sidebar.subheader("About NextTick")
        st.sidebar.info("""
        NextTick AI combines:
        - Historical price data
        - Technical indicators
        - News sentiment analysis
        - Machine learning predictions
        
        For intelligent stock market insights.
        """)
    
    def analyze_stock(self, symbol, period, use_technical_indicators, use_sentiment_analysis):
        """Main analysis function"""
        
        # Initialize session state for data
        if 'stock_data' not in st.session_state or st.session_state.get('current_symbol') != symbol:
            with st.spinner(f"Fetching data for {symbol}..."):
                # Fetch stock data
                stock_data = self.data_fetcher.get_stock_data(symbol, period)
                
                if stock_data is None or stock_data.empty:
                    st.error(f"Could not fetch data for {symbol}. Please check the symbol and try again.")
                    return
                
                # Calculate technical indicators
                if use_technical_indicators:
                    stock_data = self.data_fetcher.calculate_technical_indicators(stock_data)
                
                # Fetch news data
                news_data = self.data_fetcher.get_news_data(symbol)
                
                # Store in session state
                st.session_state.stock_data = stock_data
                st.session_state.news_data = news_data
                st.session_state.current_symbol = symbol
        
        stock_data = st.session_state.stock_data
        news_data = st.session_state.news_data
        
        # Display basic stock info
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            current_price = stock_data['Close'].iloc[-1]
            st.metric("Current Price", f"${current_price:.2f}")
        
        with col2:
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
        
        # Stock price chart
        st.subheader("Stock Price Chart")
        price_chart = self.visualizer.plot_stock_price(stock_data, f"{symbol} Stock Price")
        st.plotly_chart(price_chart, use_container_width=True)
        
        # Technical indicators
        if use_technical_indicators:
            st.subheader("Technical Indicators")
            tech_fig = self.visualizer.plot_technical_indicators(stock_data)
            st.pyplot(tech_fig)
        
        # Sentiment Analysis
        sentiment_score = 0.0
        if use_sentiment_analysis and not news_data.empty:
            st.subheader("Market Sentiment Analysis")
            
            with st.spinner("Analyzing news sentiment..."):
                sentiment_score = self.sentiment_analyzer.analyze_news_sentiment(news_data)
            
            # Display sentiment gauge
            sentiment_chart = self.visualizer.plot_sentiment_analysis(sentiment_score, news_data)
            st.plotly_chart(sentiment_chart, use_container_width=True)
            
            # Show news articles
            st.subheader("Recent News Articles")
            for idx, article in news_data.head(5).iterrows():
                with st.expander(f"{article['title']} - {article['source']}"):
                    st.write(f"**Published:** {article['published_at']}")
                    st.write(f"**Description:** {article.get('description', 'No description available')}")
        
        # Machine Learning Prediction
        st.subheader("AI Prediction & Recommendation")
        
        with st.spinner("Training model and making prediction..."):
            # Train model
            feature_columns = ['Close', 'Volume']
            if use_technical_indicators:
                feature_columns.extend(['rsi', 'macd'])
            
            # Filter out rows with NaN values in feature columns
            train_data = stock_data[feature_columns].dropna()
            
            if len(train_data) > 10:
                success = self.stock_predictor.train_linear_regression(train_data, feature_columns)
                
                if success:
                    # Make prediction
                    predicted_price, current_price = self.stock_predictor.predict_next_day(
                        train_data, feature_columns
                    )
                    
                    if predicted_price is not None:
                        # Generate recommendation
                        price_change_pred = ((predicted_price - current_price) / current_price) * 100
                        
                        # Determine recommendation
                        if price_change_pred > 1.0 and sentiment_score > 0.1:
                            recommendation = "BUY"
                            recommendation_class = "buy-recommendation"
                            recommendation_color = "green"
                        elif price_change_pred < -1.0 or sentiment_score < -0.1:
                            recommendation = "SELL"
                            recommendation_class = "sell-recommendation"
                            recommendation_color = "red"
                        else:
                            recommendation = "HOLD"
                            recommendation_class = "hold-recommendation"
                            recommendation_color = "orange"
                        
                        # Display prediction results
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Current Price", f"${current_price:.2f}")
                        
                        with col2:
                            st.metric("Predicted Price", f"${predicted_price:.2f}", 
                                     f"{price_change_pred:+.2f}%")
                        
                        with col3:
                            st.metric("Sentiment Score", f"{sentiment_score:.3f}")
                        
                        # Recommendation card
                        st.markdown(
                            f"""
                            <div class="prediction-card {recommendation_class}">
                                <h3 style="color: {recommendation_color}; margin: 0;">Recommendation: {recommendation}</h3>
                                <p style="margin: 0.5rem 0 0 0;">
                                    Based on price prediction and market sentiment analysis.
                                </p>
                            </div>
                            """, 
                            unsafe_allow_html=True
                        )
                        
                        # Prediction explanation
                        st.info(f"""
                        **Analysis Summary:**
                        - **Price Prediction:** {price_change_pred:+.2f}% expected change
                        - **Market Sentiment:** {'Positive' if sentiment_score > 0.1 else 'Negative' if sentiment_score < -0.1 else 'Neutral'}
                        - **Confidence:** Based on technical analysis and news sentiment
                        """)
            
            else:
                st.warning("Insufficient data for accurate prediction. Please select a longer time period.")
        
        # Risk Disclaimer
        st.markdown("---")
        st.warning("""
        **Disclaimer:** This is for educational and research purposes only. 
        Stock market predictions are inherently uncertain and past performance 
        does not guarantee future results. Always conduct your own research 
        and consult with financial advisors before making investment decisions.
        """)

# Run the application
if __name__ == "__main__":
    app = NextTickApp()
    app.run()
