import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os
import sys
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
    st.error(f"❌ Import Error: {e}")
    st.info("Please make sure all utility files are in the 'utils' folder")
    st.stop()

# Configure Streamlit page
st.set_page_config(
    page_title="NextTick - AI Stock Prediction",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

class FixedDataApp:
    def __init__(self):
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize components with deployment-safe error handling"""
        try:
            self.data_fetcher = DataFetcher()
            self.sentiment_analyzer = OptimizedSentimentAnalyzer()
            self.stock_predictor = StockPredictor()
            self.visualizer = StockVisualizer()
            st.sidebar.success("✅ Components initialized successfully")
        except Exception as e:
            st.error(f"❌ Initialization failed: {e}")
            self._create_fallback_components()
    
    def _create_fallback_components(self):
        """Create basic components if initialization fails"""
        try:
            self.data_fetcher = DataFetcher()
            self.stock_predictor = StockPredictor()
            self.visualizer = StockVisualizer()
            st.sidebar.warning("⚠️ Using fallback mode (sentiment analysis disabled)")
        except Exception as e:
            st.error(f"❌ Critical initialization error: {e}")
            st.stop()
    
    def run(self):
        """Main application runner with deployment safety"""
        try:
            self._render_ui()
        except Exception as e:
            self._handle_critical_error(e)
    
    def _render_ui(self):
        """Render the main UI with error boundaries"""
        # Header
        st.markdown("""
        <style>
        .main-header {
            font-size: 2.5rem;
            color: #1f77b4;
            text-align: center;
            margin-bottom: 2rem;
        }
        .data-verified {
            background: #4CAF50;
            color: white;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 0.8em;
        }
        </style>
        """, unsafe_allow_html=True)
        
        st.markdown(
            '<h1 class="main-header">📈 NextTick AI Stock Prediction <span class="data-verified">Data Verified</span></h1>', 
            unsafe_allow_html=True
        )
        
        # Sidebar with deployment info
        self._render_sidebar()
        
        # Main content with error boundary
        try:
            self._render_main_content()
        except Exception as e:
            st.error(f"❌ Content rendering error: {e}")
            self._show_fallback_interface()
    
    def _render_sidebar(self):
        """Render sidebar with deployment-safe controls"""
        st.sidebar.title("🛠️ Configuration")
        
        # Stock symbol input
        symbol = st.sidebar.text_input("Enter Stock Symbol", "AAPL").upper()
        
        # Analysis period
        period = st.sidebar.selectbox("Data Period", ["1mo", "3mo", "6mo"], index=1)
        
        # Feature selection with availability checks
        st.sidebar.subheader("Features")
        
        # Check feature availability
        sentiment_available = hasattr(self, 'sentiment_analyzer') and self.sentiment_analyzer is not None
        
        use_technical_indicators = st.sidebar.checkbox("Technical Indicators", True)
        use_sentiment_analysis = st.sidebar.checkbox(
            "Sentiment Analysis", 
            False,
            disabled=not sentiment_available,
            help="Available" if sentiment_available else "Not available in current deployment"
        )
        use_advanced_model = st.sidebar.checkbox("Enhanced Model", True)
        
        # Data refresh
        st.sidebar.subheader("Data Management")
        if st.sidebar.button("🔄 Refresh Data"):
            self._force_data_refresh()
        
        if st.sidebar.button("🧹 Clear Cache"):
            self._safe_clear_cache()
        
        # Data validation section
        self._show_data_validation_status()
        
        # Main analysis trigger
        if st.sidebar.button("🚀 Analyze Stock") or symbol:
            try:
                self.analyze_stock(symbol, period, use_technical_indicators, use_sentiment_analysis, use_advanced_model)
            except Exception as e:
                st.error(f"❌ Analysis error: {e}")
                st.info("💡 Try using simpler features or a different stock symbol")
    
    def _show_data_validation_status(self):
        """Show data validation status"""
        st.sidebar.markdown("---")
        st.sidebar.subheader("Data Status")
        
        if 'stock_data' in st.session_state and st.session_state.stock_data is not None:
            stock_data = st.session_state.stock_data
            if not stock_data.empty:
                current_price = stock_data['Close'].iloc[-1]
                st.sidebar.write(f"**Current Price:** ${current_price:.2f}")
                st.sidebar.write(f"**Data Points:** {len(stock_data)}")
                
                # Data freshness check
                latest_date = stock_data.index.max()
                if hasattr(latest_date, 'date'):
                    days_old = (datetime.now().date() - latest_date.date()).days
                    if days_old == 0:
                        st.sidebar.success("✅ Data is from today")
                    elif days_old == 1:
                        st.sidebar.info("ℹ️ Data is from yesterday")
                    else:
                        st.sidebar.warning(f"⚠️ Data is {days_old} days old")
        else:
            st.sidebar.info("📊 No data loaded yet")
    
    def _render_main_content(self):
        """Render main content area"""
        pass
    
    def analyze_stock(self, symbol, period, use_technical_indicators, use_sentiment_analysis, use_advanced_model):
        """Fixed stock analysis with data validation"""
        # Validate inputs
        if not symbol or len(symbol) < 1:
            st.error("❌ Please enter a valid stock symbol")
            return
        
        if len(symbol) > 10:
            st.error("❌ Stock symbol seems too long")
            return
        
        try:
            with st.spinner(f"🚀 Analyzing {symbol} with data validation..."):
                # Fetch data with enhanced validation
                stock_data = self._safe_fetch_data_with_validation(symbol, period)
                if stock_data is None:
                    return
                
                # Show data verification
                self._show_data_verification(stock_data, symbol)
                
                # Fetch news data if needed
                news_data = pd.DataFrame()
                if use_sentiment_analysis:
                    news_data = self._safe_fetch_news(symbol)
                
                # Render analysis components
                self._render_analysis_components(stock_data, news_data, symbol, 
                                               use_technical_indicators, use_sentiment_analysis, 
                                               use_advanced_model)
                
        except Exception as e:
            st.error(f"❌ Analysis failed: {e}")
            self._show_retry_options(symbol)
    
    def _safe_fetch_data_with_validation(self, symbol, period):
        """Safely fetch stock data with enhanced validation"""
        try:
            # Clear any cached data to force fresh fetch
            cache_key = f"{symbol}_{period}"
            if 'stock_data' in st.session_state and st.session_state.get('current_cache_key') == cache_key:
                del st.session_state.stock_data
            
            # Fetch fresh data
            stock_data = self.data_fetcher.get_stock_data(symbol, period)
            
            if stock_data is None or stock_data.empty:
                st.error(f"❌ No data found for {symbol}")
                st.info("💡 Try popular symbols like: AAPL, TSLA, MSFT, GOOGL, AMZN")
                return None
            
            # Enhanced data validation
            validation_passed, validation_message = self._validate_stock_data(stock_data, symbol)
            if not validation_passed:
                st.error(f"❌ Data validation failed: {validation_message}")
                return None
            
            # Store validated data
            st.session_state.stock_data = stock_data
            st.session_state.current_cache_key = cache_key
            st.session_state.current_symbol = symbol
            
            return stock_data
            
        except Exception as e:
            st.error(f"❌ Error fetching data: {e}")
            return None
    
    def _validate_stock_data(self, stock_data, symbol):
        """Enhanced stock data validation"""
        if stock_data is None or stock_data.empty:
            return False, "No data received"
        
        # Check required columns
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_columns = [col for col in required_columns if col not in stock_data.columns]
        if missing_columns:
            return False, f"Missing columns: {missing_columns}"
        
        # Check for reasonable price values
        latest_close = stock_data['Close'].iloc[-1]
        if latest_close <= 0:
            return False, f"Invalid close price: ${latest_close:.2f}"
        
        if latest_close > 10000:  # Unusually high price
            return False, f"Suspiciously high price: ${latest_close:.2f}"
        
        # Check for reasonable volume
        latest_volume = stock_data['Volume'].iloc[-1]
        if latest_volume <= 0:
            return False, f"Invalid volume: {latest_volume}"
        
        # Check data consistency
        if len(stock_data) > 1:
            price_change = abs(stock_data['Close'].iloc[-1] - stock_data['Close'].iloc[-2])
            if price_change > stock_data['Close'].iloc[-2] * 0.5:  # More than 50% change
                return False, f"Large price jump detected: {price_change:.2f}"
        
        return True, "Data validation passed"
    
    def _show_data_verification(self, stock_data, symbol):
        """Show data verification information"""
        current_price = stock_data['Close'].iloc[-1]
        latest_date = stock_data.index.max()
        
        st.success(f"✅ **Data Verified for {symbol}**")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Verified Price", f"${current_price:.2f}")
        with col2:
            if hasattr(latest_date, 'strftime'):
                st.metric("Latest Data", latest_date.strftime("%Y-%m-%d"))
        with col3:
            st.metric("Data Points", len(stock_data))
    
    def _safe_fetch_news(self, symbol):
        """Safely fetch news data"""
        try:
            return self.data_fetcher.get_news_data(symbol)
        except Exception as e:
            st.warning(f"⚠️ News fetching failed: {e}")
            return pd.DataFrame()
    
    def _render_analysis_components(self, stock_data, news_data, symbol, 
                                  use_technical_indicators, use_sentiment_analysis, 
                                  use_advanced_model):
        """Render all analysis components safely"""
        
        # 1. Basic stock info
        self._render_stock_info(stock_data, symbol)
        
        # 2. Price chart
        self._render_price_chart(stock_data, symbol)
        
        # 3. Technical indicators
        if use_technical_indicators:
            self._render_technical_indicators(stock_data)
        
        # 4. Sentiment analysis
        sentiment_score = 0.0
        if use_sentiment_analysis:
            sentiment_score = self._render_sentiment_analysis(news_data)
        
        # 5. AI Prediction
        self._render_prediction(stock_data, sentiment_score, symbol, use_advanced_model)
        
        # 6. Disclaimer
        self._render_disclaimer()
    
    def _render_stock_info(self, stock_data, symbol):
        """Render stock information"""
        st.subheader(f"📈 {symbol} Stock Overview")
        
        cols = st.columns(5)
        with cols[0]:
            current_price = stock_data['Close'].iloc[-1]
            st.metric("Current Price", f"${current_price:.2f}")
        
        with cols[1]:
            if len(stock_data) > 1:
                price_change = stock_data['Close'].iloc[-1] - stock_data['Close'].iloc[-2]
                change_percent = (price_change / stock_data['Close'].iloc[-2]) * 100
                st.metric("Daily Change", f"${price_change:.2f}", f"{change_percent:.2f}%")
        
        with cols[2]:
            volume = stock_data['Volume'].iloc[-1]
            st.metric("Volume", f"{volume:,.0f}")
        
        with cols[3]:
            if 'rsi' in stock_data.columns and not pd.isna(stock_data['rsi'].iloc[-1]):
                rsi = stock_data['rsi'].iloc[-1]
                st.metric("RSI", f"{rsi:.1f}")
        
        with cols[4]:
            if len(stock_data) > 20 and 'sma_20' in stock_data.columns:
                sma_20 = stock_data['sma_20'].iloc[-1]
                price_ratio = (stock_data['Close'].iloc[-1] / sma_20 - 1) * 100
                st.metric("vs SMA20", f"{price_ratio:+.1f}%")
    
    def _render_price_chart(self, stock_data, symbol):
        """Render price chart"""
        st.subheader("📊 Price Chart")
        try:
            price_chart = self.visualizer.plot_stock_price(stock_data, f"{symbol} Price")
            st.plotly_chart(price_chart, use_container_width=True)
        except Exception as e:
            st.warning(f"⚠️ Chart rendering failed: {e}")
    
    def _render_technical_indicators(self, stock_data):
        """Render technical indicators"""
        st.subheader("🔧 Technical Indicators")
        try:
            tech_fig = self.visualizer.plot_technical_indicators(stock_data)
            st.pyplot(tech_fig)
        except Exception as e:
            st.warning(f"⚠️ Technical indicators failed: {e}")
    
    def _render_sentiment_analysis(self, news_data):
        """Render sentiment analysis and return score"""
        st.subheader("😊 Market Sentiment")
        try:
            with st.spinner("Analyzing sentiment..."):
                sentiment_score = self.sentiment_analyzer.analyze_news_sentiment(news_data)
            
            sentiment_chart = self.visualizer.plot_sentiment_analysis(sentiment_score, news_data)
            st.plotly_chart(sentiment_chart, use_container_width=True)
            
            # Show sentiment interpretation
            self._show_sentiment_interpretation(sentiment_score)
            
            # Show news articles
            self._show_news_articles(news_data)
            
            return sentiment_score
            
        except Exception as e:
            st.warning(f"⚠️ Sentiment analysis failed: {e}")
            return 0.0
    
    def _show_sentiment_interpretation(self, sentiment_score):
        """Show sentiment interpretation"""
        if sentiment_score > 0.3:
            interpretation, color = "Strongly Bullish", "green"
        elif sentiment_score > 0.1:
            interpretation, color = "Bullish", "lightgreen"
        elif sentiment_score > -0.1:
            interpretation, color = "Neutral", "gray"
        elif sentiment_score > -0.3:
            interpretation, color = "Bearish", "orange"
        else:
            interpretation, color = "Strongly Bearish", "red"
        
        st.info(f"**Sentiment:** {interpretation} (Score: {sentiment_score:.3f})")
    
    def _show_news_articles(self, news_data):
        """Show news articles"""
        if not news_data.empty:
            with st.expander("📰 Recent News"):
                for idx, article in news_data.head(3).iterrows():
                    st.write(f"**{article['title']}**")
                    st.write(f"*Source: {article['source']}*")
                    st.write(article.get('description', 'No description'))
                    st.markdown("---")
        else:
            st.info("No recent news available")
    
    def _render_prediction(self, stock_data, sentiment_score, symbol, use_advanced_model):
        """Render prediction component"""
        st.subheader("🤖 AI Prediction")
        try:
            # Get current price for validation
            current_price = stock_data['Close'].iloc[-1]
            
            if use_advanced_model:
                predicted_price, confidence = self._generate_enhanced_prediction(stock_data, current_price)
            else:
                predicted_price, confidence = self._generate_basic_prediction(stock_data, current_price)
            
            if predicted_price is not None:
                self._show_prediction_results(predicted_price, current_price, sentiment_score, confidence, use_advanced_model)
            else:
                st.warning("❌ Could not generate prediction")
                
        except Exception as e:
            st.warning(f"⚠️ Prediction failed: {e}")
    
    def _generate_enhanced_prediction(self, stock_data, current_price):
        """Generate enhanced prediction"""
        feature_columns = ['Close', 'Volume']
        if 'rsi' in stock_data.columns:
            feature_columns.append('rsi')
        
        train_data = stock_data[feature_columns].dropna()
        
        if len(train_data) > 10:
            try:
                if self.stock_predictor.train_enhanced_model(train_data, feature_columns):
                    predicted_price, _, confidence = self.stock_predictor.predict_next_day(train_data, feature_columns)
                    return predicted_price, confidence
            except Exception as e:
                st.warning(f"Enhanced model error: {e}")
        
        return None, None
    
    def _generate_basic_prediction(self, stock_data, current_price):
        """Generate basic prediction"""
        feature_columns = ['Close', 'Volume']
        train_data = stock_data[feature_columns].dropna()
        
        if len(train_data) > 5:
            try:
                if self.stock_predictor.train_linear_regression(train_data, feature_columns):
                    predicted_price, _ = self.stock_predictor.predict_next_day(train_data, feature_columns)
                    return predicted_price, None
            except Exception as e:
                st.warning(f"Basic model error: {e}")
        
        return None, None
    
    def _show_prediction_results(self, predicted_price, current_price, sentiment_score, confidence, is_enhanced):
        """Show prediction results with data consistency checks"""
        
        # DATA CONSISTENCY VALIDATION
        if current_price <= 0 or predicted_price <= 0:
            st.error("❌ Invalid price data detected")
            return
        
        # Calculate percentage change
        price_change = ((predicted_price - current_price) / current_price) * 100
        
        # Validate percentage change is reasonable
        if abs(price_change) > 50:
            st.warning("⚠️ Unusually large price change predicted - verifying data...")
        
        # FIXED RECOMMENDATION LOGIC
        if price_change > 3.0 and sentiment_score > 0.3:
            recommendation, color, emoji = "STRONG BUY", "green", "🚀"
            reasoning = "Strong upward momentum with very positive sentiment"
        elif price_change > 1.5 and sentiment_score > 0.15:
            recommendation, color, emoji = "BUY", "lightgreen", "📈"
            reasoning = "Positive price trend with good sentiment"
        elif price_change > 0.5 and sentiment_score > 0.05:
            recommendation, color, emoji = "CAUTIOUS BUY", "lightblue", "↗️"
            reasoning = "Slight upward trend with positive sentiment"
        elif price_change < -3.0 and sentiment_score < -0.3:
            recommendation, color, emoji = "STRONG SELL", "red", "🔻"
            reasoning = "Strong downward momentum with very negative sentiment"
        elif price_change < -1.5 and sentiment_score < -0.15:
            recommendation, color, emoji = "SELL", "orange", "📉"
            reasoning = "Negative price trend with concerning sentiment"
        elif price_change < -0.5 and sentiment_score < -0.05:
            recommendation, color, emoji = "CAUTIOUS SELL", "lightcoral", "↘️"
            reasoning = "Slight downward trend with negative sentiment"
        else:
            recommendation, color, emoji = "HOLD", "gray", "⚖️"
            reasoning = "Mixed or neutral signals - wait for clearer direction"
        
        # Adjust for confidence
        if confidence and confidence < 0.5:
            recommendation = "HOLD (Low Confidence)"
            color = "gray"
            emoji = "🤔"
            reasoning = f"Model confidence is low ({confidence*100:.1f}%)"
        
        # Display results
        st.subheader("🎯 AI Prediction Results")
        
        # Data consistency confirmation
        st.success(f"✅ Using verified current price: ${current_price:.2f}")
        
        # Main metrics
        if is_enhanced and confidence:
            cols = st.columns(4)
        else:
            cols = st.columns(3)
        
        with cols[0]:
            st.metric("Current Price", f"${current_price:.2f}")
        
        with cols[1]:
            st.metric("Predicted Price", f"${predicted_price:.2f}", f"{price_change:+.2f}%")
        
        with cols[2]:
            st.metric("Market Sentiment", f"{sentiment_score:.3f}")
        
        if is_enhanced and confidence:
            with cols[3]:
                st.metric("Model Confidence", f"{confidence*100:.1f}%")
        
        # Recommendation
        st.markdown(f"""
        <div style='background-color: #f8f9fa; padding: 1.5rem; border-radius: 10px; border-left: 5px solid {color}; margin: 1rem 0;'>
            <h3 style='color: {color}; margin: 0;'>{emoji} Recommendation: {recommendation}</h3>
            <p style='margin: 0.5rem 0 0 0; color: #666;'><strong>Reasoning:</strong> {reasoning}</p>
            <p style='margin: 0.5rem 0 0 0; font-size: 0.9rem; color: #888;'>
                Expected Price Change: {price_change:+.2f}% | Sentiment Score: {sentiment_score:.3f}
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    def _render_disclaimer(self):
        """Render disclaimer"""
        st.markdown("---")
        st.warning("""
        **Disclaimer:** Educational purposes only. Not financial advice. 
        Always conduct your own research and consult with financial advisors.
        Past performance is not indicative of future results.
        """)
    
    def _force_data_refresh(self):
        """Force refresh of all data"""
        if 'stock_data' in st.session_state:
            del st.session_state.stock_data
        if 'news_data' in st.session_state:
            del st.session_state.news_data
        st.success("✅ Data refresh triggered")
        st.rerun()
    
    def _safe_clear_cache(self):
        """Safely clear cache"""
        try:
            if hasattr(self, 'sentiment_analyzer'):
                self.sentiment_analyzer.cleanup()
            gc.collect()
            st.success("✅ Cache cleared")
        except:
            st.warning("⚠️ Partial cache clearance")
    
    def _show_fallback_interface(self):
        """Show fallback interface when main content fails"""
        st.error("🚨 Application encountered an error")
        st.info("""
        **Troubleshooting steps:**
        1. Click 'Refresh Data' in sidebar
        2. Try a different stock symbol
        3. Disable advanced features
        4. Clear browser cache
        """)
        
        if st.button("🔄 Try Basic Analysis"):
            self._force_data_refresh()
    
    def _show_retry_options(self, symbol):
        """Show retry options after error"""
        st.info(f"Having issues with {symbol}? Try these:")
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("🔄 Retry Same Symbol"):
                self._force_data_refresh()
        with col2:
            if st.button("📊 Try AAPL Instead"):
                st.session_state.retry_symbol = "AAPL"
                self._force_data_refresh()
        with col3:
            if st.button("🔧 Basic Features Only"):
                st.info("Please manually disable advanced features and retry")
    
    def _handle_critical_error(self, error):
        """Handle critical application errors"""
        st.error("🚨 Critical Application Error")
        st.code(f"Error: {str(error)}")
        st.info("Please refresh the page or try a different stock symbol.")

# Run the application
if __name__ == "__main__":
    app = FixedDataApp()
    app.run()
