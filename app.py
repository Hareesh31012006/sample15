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
try:
    from utils.data_fetcher import DataFetcher
    from utils.sentiment_analyzer import OptimizedSentimentAnalyzer
    from utils.model import StockPredictor
    from utils.visualization import StockVisualizer
except ImportError as e:
    st.error(f"Error importing modules: {e}")
    st.stop()

# Load environment variables
load_dotenv()

# Configure Streamlit page
st.set_page_config(
    page_title="NextTick - AI Stock Prediction",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

class NextTickApp:
    def __init__(self):
        self.data_fetcher = DataFetcher()
        self.sentiment_analyzer = OptimizedSentimentAnalyzer()
        self.stock_predictor = StockPredictor()
        self.visualizer = StockVisualizer()
    
    def run(self):
        # Header
        st.markdown("""
        <style>
        .main-header {
            font-size: 2.5rem;
            color: #1f77b4;
            text-align: center;
            margin-bottom: 2rem;
        }
        </style>
        """, unsafe_allow_html=True)
        
        st.markdown('<h1 class="main-header">📈 NextTick AI Stock Prediction</h1>', unsafe_allow_html=True)
        
        # Sidebar
        st.sidebar.title("Configuration")
        
        # Stock symbol input
        symbol = st.sidebar.text_input("Enter Stock Symbol", "TSLA").upper()
        
        # Analysis period
        period = st.sidebar.selectbox("Data Period", ["1mo", "3mo", "6mo", "1y"], index=1)
        
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
        
        # Main content
        if st.sidebar.button("Analyze Stock") or symbol:
            try:
                self.analyze_stock(symbol, period, use_technical_indicators, use_sentiment_analysis, use_advanced_model)
            except Exception as e:
                st.error(f"Application error: {str(e)}")
                st.info("💡 Try clearing the memory cache or using a shorter data period.")
    
    def _clear_memory_cache(self):
        """Clear memory cache"""
        try:
            if hasattr(self, 'sentiment_analyzer') and self.sentiment_analyzer:
                self.sentiment_analyzer.cleanup()
            
            # Clear session state
            for key in list(st.session_state.keys()):
                if key not in ['current_cache_key', 'current_symbol']:
                    del st.session_state[key]
            
            gc.collect()
            st.success("🔄 Memory cache cleared!")
            
        except Exception as e:
            st.warning(f"Memory cleanup note: {e}")
    
    def analyze_stock(self, symbol, period, use_technical_indicators, use_sentiment_analysis, use_advanced_model):
        """Main stock analysis function"""
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
                    with st.spinner("🤖 Analyzing news sentiment with advanced model..."):
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
            **Disclaimer:** This tool is for educational and research purposes only. 
            Stock predictions are based on AI models and historical data, which may not accurately predict future performance. 
            Always conduct your own research and consult with financial advisors before making investment decisions.
            Past performance is not indicative of future results.
            """)
            
        except Exception as e:
            st.error(f"Analysis error: {str(e)}")
    
    def _display_stock_info(self, stock_data, symbol):
        """Display stock information"""
        st.subheader(f"📈 {symbol} Stock Overview")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
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
            if 'rsi' in stock_data.columns and not pd.isna(stock_data['rsi'].iloc[-1]):
                rsi = stock_data['rsi'].iloc[-1]
                st.metric("RSI", f"{rsi:.1f}")
        
        with col5:
            if len(stock_data) > 20 and 'sma_20' in stock_data.columns:
                sma_20 = stock_data['sma_20'].iloc[-1]
                price_ratio = (stock_data['Close'].iloc[-1] / sma_20 - 1) * 100
                st.metric("vs SMA20", f"{price_ratio:+.1f}%")
    
    def _display_sentiment_interpretation(self, sentiment_score):
        """Display sentiment interpretation"""
        if sentiment_score > 0.3:
            interpretation = "Strongly Bullish"
            color = "green"
            emoji = "🚀"
        elif sentiment_score > 0.1:
            interpretation = "Moderately Bullish" 
            color = "lightgreen"
            emoji = "📈"
        elif sentiment_score > -0.1:
            interpretation = "Neutral"
            color = "gray"
            emoji = "➡️"
        elif sentiment_score > -0.3:
            interpretation = "Moderately Bearish"
            color = "orange"
            emoji = "📉"
        else:
            interpretation = "Strongly Bearish"
            color = "red"
            emoji = "🔻"
        
        st.markdown(f"""
        <div style='background-color: #f8f9fa; padding: 1rem; border-radius: 8px; border-left: 4px solid {color}; margin: 1rem 0;'>
            <h4 style='color: {color}; margin: 0;'>{emoji} Sentiment: {interpretation}</h4>
            <p style='margin: 0.5rem 0 0 0; color: #666;'>Score: {sentiment_score:.3f}</p>
        </div>
        """, unsafe_allow_html=True)
    
    def _display_news_articles(self, news_data):
        """Display news articles"""
        st.subheader("📰 Recent News Articles")
        if not news_data.empty:
            for idx, article in news_data.head(4).iterrows():
                with st.expander(f"{article['title']} - {article['source']}"):
                    try:
                        pub_date = pd.to_datetime(article['published_at']).strftime("%Y-%m-%d %H:%M")
                    except:
                        pub_date = str(article['published_at'])
                    
                    st.write(f"**Published:** {pub_date}")
                    st.write(f"**Description:** {article.get('description', 'No description available')}")
        else:
            st.info("No recent news articles available.")
    
    def _generate_enhanced_prediction(self, stock_data, sentiment_score, symbol):
        """Enhanced prediction with advanced model"""
        with st.spinner("🧠 Training enhanced model and making prediction..."):
            # Use enhanced feature set
            feature_columns = ['Close', 'Volume']
            if 'rsi' in stock_data.columns:
                feature_columns.append('rsi')
            if 'macd' in stock_data.columns:
                feature_columns.append('macd')
            if 'sma_20' in stock_data.columns:
                feature_columns.append('sma_20')
            
            # Filter out rows with NaN values
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
                        st.warning("Enhanced model training failed. Using basic model...")
                        self._generate_prediction(stock_data, sentiment_score, symbol)
                        
                except Exception as e:
                    st.warning(f"Enhanced prediction model error: {e}")
                    self._generate_prediction(stock_data, sentiment_score, symbol)
            else:
                st.warning("Insufficient data for enhanced prediction. Using basic model...")
                self._generate_prediction(stock_data, sentiment_score, symbol)
    
    def _generate_prediction(self, stock_data, sentiment_score, symbol):
        """Basic stock prediction"""
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
                recommendation = f"HOLD (LOW CONFIDENCE)"
                recommendation_color = "gray"
                recommendation_emoji = "🤔"
                reasoning = f"Model confidence is low ({confidence*100:.1f}%)"
        
        # SPECIAL CASES that always result in HOLD
        if (abs(price_change_pred) < 0.5 and abs(sentiment_score) < 0.1):
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
            st.metric("Sentiment Score", f"{sentiment_score:.3f}")
        
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
        </div>
        """, unsafe_allow_html=True)
    
    def _display_prediction_results(self, predicted_price, current_price, sentiment_score):
        """Display basic prediction results"""
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
                        st.metric("R² Score", f"{r2:.3f}")
                    
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
    app = NextTickApp()
    app.run()
