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
    
    def _smart_memory_management(self):
        """Smart memory management that doesn't affect accuracy"""
        if 'analysis_count' not in st.session_state:
            st.session_state.analysis_count = 0
            st.session_state.last_memory_clear = datetime.now()
        
        st.session_state.analysis_count += 1
        
        # Clear memory only after 4 analyses or 5 minutes (increased from 3)
        current_time = datetime.now()
        time_since_clear = (current_time - st.session_state.last_memory_clear).total_seconds()
        
        if st.session_state.analysis_count >= 4 or time_since_clear > 300:  # 5 minutes
            st.session_state.analysis_count = 0
            st.session_state.last_memory_clear = current_time
            self._clear_memory_cache()
    
    def _clear_memory_cache(self):
        """Enhanced memory clearing without losing accuracy"""
        import gc
        try:
            # Clear sentiment analyzer memory
            if self.sentiment_analyzer:
                self.sentiment_analyzer.cleanup()
            
            # Clear large data objects but keep current analysis
            current_data = {}
            if 'stock_data' in st.session_state:
                current_data['stock_data'] = st.session_state.stock_data
            if 'news_data' in st.session_state:
                current_data['news_data'] = st.session_state.news_data
            if 'current_cache_key' in st.session_state:
                current_data['current_cache_key'] = st.session_state.current_cache_key
            if 'current_symbol' in st.session_state:
                current_data['current_symbol'] = st.session_state.current_symbol
            
            # Clear all session state
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            
            # Restore current analysis data
            for key, value in current_data.items():
                st.session_state[key] = value
            
            # Force garbage collection
            gc.collect()
            
            st.success("🔄 Memory optimized! Model accuracy maintained.")
            
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
            margin-bottom: 2rem;
        }
        .metric-card {
            background-color: #f0f2f6;
            padding: 1rem;
            border-radius: 10px;
            border-left: 4px solid #1f77b4;
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
        
        # Display memory info
        if 'analysis_count' in st.session_state:
            st.sidebar.info(f"Analyses since last clear: {st.session_state.analysis_count}/4")
        
        # Main content
        if st.sidebar.button("Analyze Stock") or symbol:
            try:
                self.analyze_stock(symbol, period, use_technical_indicators, use_sentiment_analysis, use_advanced_model)
            except Exception as e:
                self._handle_error(e)
        
        # About section
        st.sidebar.markdown("---")
        st.sidebar.info("""
        **Enhanced Accuracy Features:**
        - Advanced sentiment analysis (4 articles)
        - Enhanced feature engineering
        - Confidence-based predictions
        - Smart memory management
        """)
    
    def _handle_error(self, error):
        """Handle errors with memory cleanup"""
        st.error(f"Application error: {str(error)}")
        st.info("💡 Try clearing the memory cache or using a shorter data period.")
        
        # Force cleanup on error
        self._clear_memory_cache()
    
    def analyze_stock(self, symbol, period, use_technical_indicators, use_sentiment_analysis, use_advanced_model):
        """Optimized stock analysis with smart memory management"""
        try:
            # Use smart memory management
            self._smart_memory_management()
            
            # Initialize session state for data
            cache_key = f"{symbol}_{period}"
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
            
            # Sentiment Analysis (with memory optimization)
            sentiment_score = 0.0
            if use_sentiment_analysis:
                st.subheader("😊 Market Sentiment Analysis")
                try:
                    with st.spinner("🤖 Analyzing news sentiment with enhanced model..."):
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
            self._handle_error(e)
    
    def _display_stock_info(self, stock_data, symbol):
        """Display enhanced stock information"""
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
                rsi_color = "red" if rsi > 70 else "green" if rsi < 30 else "orange"
                st.metric("RSI", f"{rsi:.1f}", delta_color="off")
        
        with col5:
            # Price range for the period
            price_range = (stock_data['High'].max() - stock_data['Low'].min()) / stock_data['Close'].mean() * 100
            st.metric("Volatility", f"{price_range:.1f}%")
    
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
            for idx, article in news_data.head(4).iterrows():  # Show 4 articles for better context
                with st.expander(f"{article['title']} - {article['source']}"):
                    # Format published date
                    try:
                        pub_date = pd.to_datetime(article['published_at']).strftime("%Y-%m-%d %H:%M")
                    except:
                        pub_date = str(article['published_at'])
                    
                    st.write(f"**Published:** {pub_date}")
                    st.write(f"**Description:** {article.get('description', 'No description available')}")
                    
                    # Add sentiment for this article
                    try:
                        article_text = f"{article['title']}. {article.get('description', '')}"
                        article_sentiment = self.sentiment_analyzer.analyze_with_textblob(article_text)
                        sentiment_emoji = "😊" if article_sentiment > 0.1 else "😐" if article_sentiment > -0.1 else "😞"
                        st.write(f"**Article Sentiment:** {sentiment_emoji} ({article_sentiment:.3f})")
                    except:
                        pass
        else:
            st.info("No recent news articles available.")
    
    def _generate_enhanced_prediction(self, stock_data, sentiment_score, symbol):
        """Enhanced stock prediction with better accuracy"""
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
            
            if len(train_data) > 15:  # Increased minimum data requirement
                try:
                    # Use enhanced model training
                    success = self.stock_predictor.train_enhanced_model(train_data, feature_columns)
                    
                    if success:
                        # Get prediction with confidence
                        predicted_price, current_price, confidence = self.stock_predictor.predict_next_day(train_data, feature_columns)
                        
                        if predicted_price is not None:
                            self._display_enhanced_prediction_results(
                                predicted_price, current_price, sentiment_score, confidence, symbol
                            )
                            
                            # Show model metrics
                            self._display_model_metrics(train_data, feature_columns)
                    else:
                        st.warning("Model training failed. Try with more data.")
                        
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
        """Display enhanced prediction results with confidence"""
        price_change_pred = ((predicted_price - current_price) / current_price) * 100
        
        # Enhanced recommendation algorithm with improved logic
        sentiment_weight = 0.4  # Increased sentiment weight
        price_weight = 0.6     # Reduced price weight
        
        # Combined score - improved calculation
        price_component = price_change_pred * price_weight
        sentiment_component = sentiment_score * 50 * sentiment_weight  # Scale sentiment appropriately
        
        combined_score = price_component + sentiment_component
        
        # Enhanced recommendation logic
        if price_change_pred > 0 and sentiment_score > 0:
            # Bullish scenario: price up + positive sentiment
            if price_change_pred > 2.0 and sentiment_score > 0.2:
                recommendation = "STRONG BUY"
                recommendation_color = "green"
                recommendation_emoji = "🚀"
            elif price_change_pred > 0.5 and sentiment_score > 0.05:
                recommendation = "BUY"
                recommendation_color = "lightgreen"
                recommendation_emoji = "📈"
            else:
                recommendation = "HOLD"
                recommendation_color = "orange"
                recommendation_emoji = "⚖️"
                
        elif price_change_pred < 0 and sentiment_score < 0:
            # Bearish scenario: price down + negative sentiment
            if price_change_pred < -2.0 and sentiment_score < -0.2:
                recommendation = "STRONG SELL"
                recommendation_color = "red"
                recommendation_emoji = "🔻"
            elif price_change_pred < -0.5 and sentiment_score < -0.05:
                recommendation = "SELL"
                recommendation_color = "red"
                recommendation_emoji = "📉"
            else:
                recommendation = "HOLD"
                recommendation_color = "orange"
                recommendation_emoji = "⚖️"
                
        else:
            # Mixed signals: price and sentiment going opposite directions
            if abs(price_change_pred) > abs(sentiment_score * 100):
                # Price trend dominates
                if price_change_pred > 1.0:
                    recommendation = "CAUTIOUS BUY"
                    recommendation_color = "lightgreen"
                    recommendation_emoji = "📈"
                elif price_change_pred < -1.0:
                    recommendation = "CAUTIOUS SELL"
                    recommendation_color = "orange"
                    recommendation_emoji = "📉"
                else:
                    recommendation = "HOLD"
                    recommendation_color = "orange"
                    recommendation_emoji = "⚖️"
            else:
                # Sentiment dominates
                if sentiment_score > 0.1:
                    recommendation = "SENTIMENT BUY"
                    recommendation_color = "lightgreen"
                    recommendation_emoji = "😊"
                elif sentiment_score < -0.1:
                    recommendation = "SENTIMENT SELL"
                    recommendation_color = "orange"
                    recommendation_emoji = "😞"
                else:
                    recommendation = "HOLD"
                    recommendation_color = "orange"
                    recommendation_emoji = "⚖️"
        
        # Override if confidence is very low
        if confidence and confidence < 0.3:
            recommendation = "HOLD (LOW CONFIDENCE)"
            recommendation_color = "gray"
            recommendation_emoji = "🤔"
        
        # Display enhanced results
        st.subheader("🎯 Prediction Results")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Current Price", f"${current_price:.2f}")
        
        with col2:
            st.metric("Predicted Price", f"${predicted_price:.2f}", f"{price_change_pred:+.2f}%")
        
        with col3:
            # Color code sentiment
            sentiment_color = "red" if sentiment_score < -0.1 else "green" if sentiment_score > 0.1 else "orange"
            st.metric("Sentiment Score", f"{sentiment_score:.3f}", delta_color="off")
        
        with col4:
            confidence_display = confidence * 100 if confidence else "N/A"
            if confidence:
                confidence_color = "green" if confidence > 0.7 else "orange" if confidence > 0.5 else "red"
                st.metric("Model Confidence", f"{confidence_display:.1f}%", delta_color="off")
            else:
                st.metric("Model Confidence", "N/A")
        
        # Enhanced recommendation with detailed reasoning
        reasoning = []
        
        if price_change_pred > 0:
            reasoning.append(f"price expected to rise (+{price_change_pred:.2f}%)")
        elif price_change_pred < 0:
            reasoning.append(f"price expected to fall ({price_change_pred:+.2f}%)")
        else:
            reasoning.append("price expected to remain stable")
        
        if sentiment_score > 0.1:
            reasoning.append("positive market sentiment")
        elif sentiment_score < -0.1:
            reasoning.append("negative market sentiment")
        else:
            reasoning.append("neutral market sentiment")
        
        if confidence:
            if confidence > 0.7:
                reasoning.append("high model confidence")
            elif confidence > 0.5:
                reasoning.append("moderate model confidence")
            else:
                reasoning.append("low model confidence")
        
        reasoning_text = ", ".join(reasoning)
        
        st.markdown(f"""
        <div style='background-color: #f8f9fa; padding: 1.5rem; border-radius: 10px; border-left: 5px solid {recommendation_color}; margin: 1rem 0;'>
            <h3 style='color: {recommendation_color}; margin: 0;'>{recommendation_emoji} Recommendation: {recommendation}</h3>
            <p style='margin: 0.5rem 0 0 0; color: #666;'>
                Based on analysis of: {reasoning_text}
            </p>
            <p style='margin: 0.5rem 0 0 0; font-size: 0.9rem; color: #888;'>
                Combined Score: {combined_score:.2f} | Price Impact: {price_change_pred:+.2f}% | Sentiment Impact: {sentiment_score:.3f}
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Add explanation for conflicting signals
        if (price_change_pred > 0 and sentiment_score < -0.1) or (price_change_pred < 0 and sentiment_score > 0.1):
            st.info("""
            **Note:** Mixed signals detected. The price prediction and market sentiment are pointing in different directions. 
            Consider waiting for clearer market conditions or conducting additional research.
            """)
    
    def _display_prediction_results(self, predicted_price, current_price, sentiment_score):
        """Display basic prediction results"""
        price_change_pred = ((predicted_price - current_price) / current_price) * 100
        
        # Improved basic recommendation algorithm
        if price_change_pred > 1.0 and sentiment_score > 0.1:
            recommendation = "BUY"
            recommendation_color = "green"
            recommendation_emoji = "📈"
        elif price_change_pred < -1.0 and sentiment_score < -0.1:
            recommendation = "SELL"
            recommendation_color = "red"
            recommendation_emoji = "📉"
        elif price_change_pred > 2.0:
            recommendation = "BUY (Strong Price Momentum)"
            recommendation_color = "green"
            recommendation_emoji = "🚀"
        elif price_change_pred < -2.0:
            recommendation = "SELL (Strong Price Decline)"
            recommendation_color = "red"
            recommendation_emoji = "🔻"
        elif sentiment_score > 0.2:
            recommendation = "BUY (Positive Sentiment)"
            recommendation_color = "lightgreen"
            recommendation_emoji = "😊"
        elif sentiment_score < -0.2:
            recommendation = "SELL (Negative Sentiment)"
            recommendation_color = "orange"
            recommendation_emoji = "😞"
        else:
            recommendation = "HOLD"
            recommendation_color = "orange"
            recommendation_emoji = "⚖️"
        
        # Display results
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Current Price", f"${current_price:.2f}")
        
        with col2:
            st.metric("Predicted Price", f"${predicted_price:.2f}", f"{price_change_pred:+.2f}%")
        
        with col3:
            st.metric("Sentiment Score", f"{sentiment_score:.3f}")
        
        # Basic recommendation
        st.markdown(f"""
        <div style='background-color: #f0f2f6; padding: 1.5rem; border-radius: 10px; border-left: 5px solid {recommendation_color}; margin: 1rem 0;'>
            <h3 style='color: {recommendation_color}; margin: 0;'>{recommendation_emoji} Recommendation: {recommendation}</h3>
            <p style='margin: 0.5rem 0 0 0;'>Based on AI prediction and market sentiment analysis</p>
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
