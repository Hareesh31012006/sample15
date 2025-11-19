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

# Add error handling for imports
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

class DeploymentReadyApp:
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
            # Create fallback components
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
        .deployment-badge {
            background: #4CAF50;
            color: white;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 0.8em;
        }
        </style>
        """, unsafe_allow_html=True)
        
        st.markdown(
            '<h1 class="main-header">📈 NextTick AI Stock Prediction <span class="deployment-badge">Deployment Ready</span></h1>', 
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
            False,  # Default to false for stability
            disabled=not sentiment_available,
            help="Available" if sentiment_available else "Not available in current deployment"
        )
        use_advanced_model = st.sidebar.checkbox("Enhanced Model", True)
        
        # Deployment controls
        st.sidebar.subheader("Deployment")
        if st.sidebar.button("🔄 Restart Session"):
            self._restart_session()
        
        if st.sidebar.button("🧹 Clear Cache"):
            self._safe_clear_cache()
        
        # Deployment status
        self._show_deployment_status()
        
        # Main analysis trigger
        if st.sidebar.button("🚀 Analyze Stock") or symbol:
            try:
                self.analyze_stock(symbol, period, use_technical_indicators, use_sentiment_analysis, use_advanced_model)
            except Exception as e:
                st.error(f"❌ Analysis error: {e}")
                st.info("💡 Try using simpler features or a different stock symbol")
    
    def _show_deployment_status(self):
        """Show deployment environment status"""
        st.sidebar.markdown("---")
        st.sidebar.subheader("Deployment Status")
        
        # Platform detection
        try:
            # Common deployment platform environment variables
            platform = "Unknown"
            if 'STREAMLIT_SERVER_PORT' in os.environ:
                platform = "Streamlit Cloud"
            elif 'HEROKU_APP_NAME' in os.environ:
                platform = "Heroku"
            elif 'VERCEL' in os.environ:
                platform = "Vercel"
            elif 'RENDER' in os.environ:
                platform = "Render"
            else:
                platform = "Local"
            
            st.sidebar.write(f"**Platform:** {platform}")
        except:
            st.sidebar.write("**Platform:** Local")
        
        # Feature availability
        features = []
        if hasattr(self, 'data_fetcher'):
            features.append("📊 Data Fetching")
        if hasattr(self, 'sentiment_analyzer') and self.sentiment_analyzer:
            features.append("😊 Sentiment Analysis")
        if hasattr(self, 'stock_predictor'):
            features.append("🤖 AI Prediction")
        
        st.sidebar.write("**Available Features:**")
        for feature in features:
            st.sidebar.write(f"  {feature}")
    
    def _render_main_content(self):
        """Render main content area"""
        # This will be populated by analyze_stock
        pass
    
    def analyze_stock(self, symbol, period, use_technical_indicators, use_sentiment_analysis, use_advanced_model):
        """Deployment-safe stock analysis"""
        # Validate inputs
        if not symbol or len(symbol) < 1:
            st.error("❌ Please enter a valid stock symbol")
            return
        
        if len(symbol) > 10:  # Reasonable symbol length
            st.error("❌ Stock symbol seems too long")
            return
        
        try:
            with st.spinner(f"🚀 Analyzing {symbol}..."):
                # Fetch data with timeout protection
                stock_data = self._safe_fetch_data(symbol, period)
                if stock_data is None:
                    return
                
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
    
    def _safe_fetch_data(self, symbol, period):
        """Safely fetch stock data with error handling"""
        try:
            # Set timeout for data fetching
            import signal
            class TimeoutError(Exception):
                pass
            
            def timeout_handler(signum, frame):
                raise TimeoutError("Data fetching timed out")
            
            # Set timeout (not available on all platforms)
            try:
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(30)  # 30 second timeout
                data = self.data_fetcher.get_stock_data(symbol, period)
                signal.alarm(0)  # Cancel timeout
            except:
                data = self.data_fetcher.get_stock_data(symbol, period)  # Fallback
            
            if data is None or data.empty:
                st.error(f"❌ No data found for {symbol}")
                st.info("💡 Try a different symbol like AAPL, TSLA, or MSFT")
                return None
            
            return data
            
        except TimeoutError:
            st.error("⏰ Data fetching timed out. Please try again.")
            return None
        except Exception as e:
            st.error(f"❌ Error fetching data: {e}")
            return None
    
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
        st.subheader(f"📈 {symbol} Overview")
        
        cols = st.columns(4)
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
            if 'sma_20' in stock_data.columns:
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
            if use_advanced_model:
                self._generate_enhanced_prediction(stock_data, sentiment_score, symbol)
            else:
                self._generate_basic_prediction(stock_data, sentiment_score, symbol)
        except Exception as e:
            st.warning(f"⚠️ Prediction failed: {e}")
    
    def _generate_enhanced_prediction(self, stock_data, sentiment_score, symbol):
        """Generate enhanced prediction"""
        with st.spinner("Training advanced model..."):
            feature_columns = ['Close', 'Volume']
            if 'rsi' in stock_data.columns:
                feature_columns.append('rsi')
            
            train_data = stock_data[feature_columns].dropna()
            
            if len(train_data) > 10:
                try:
                    if self.stock_predictor.train_enhanced_model(train_data, feature_columns):
                        predicted_price, current_price, confidence = self.stock_predictor.predict_next_day(train_data, feature_columns)
                        if predicted_price is not None:
                            self._show_prediction_results(predicted_price, current_price, sentiment_score, confidence, True)
                            return
                except:
                    pass
            
            # Fallback to basic prediction
            self._generate_basic_prediction(stock_data, sentiment_score, symbol)
    
    def _generate_basic_prediction(self, stock_data, sentiment_score, symbol):
        """Generate basic prediction"""
        with st.spinner("Training model..."):
            feature_columns = ['Close', 'Volume']
            train_data = stock_data[feature_columns].dropna()
            
            if len(train_data) > 5:
                try:
                    if self.stock_predictor.train_linear_regression(train_data, feature_columns):
                        predicted_price, current_price = self.stock_predictor.predict_next_day(train_data, feature_columns)
                        if predicted_price is not None:
                            self._show_prediction_results(predicted_price, current_price, sentiment_score, None, False)
                            return
                except:
                    pass
            
            st.warning("Insufficient data for prediction")
    
    def _show_prediction_results(self, predicted_price, current_price, sentiment_score, confidence, is_enhanced):
        """Show prediction results"""
        price_change = ((predicted_price - current_price) / current_price) * 100
        
        # Simple, reliable recommendation logic
        if price_change > 2.0 and sentiment_score > 0.2:
            recommendation, color = "BUY", "green"
        elif price_change > 1.0 and sentiment_score > 0.1:
            recommendation, color = "CAUTIOUS BUY", "lightgreen"
        elif price_change < -2.0 and sentiment_score < -0.2:
            recommendation, color = "SELL", "red"
        elif price_change < -1.0 and sentiment_score < -0.1:
            recommendation, color = "CAUTIOUS SELL", "orange"
        else:
            recommendation, color = "HOLD", "gray"
        
        # Display results
        cols = st.columns(4 if is_enhanced else 3)
        
        with cols[0]:
            st.metric("Current", f"${current_price:.2f}")
        with cols[1]:
            st.metric("Predicted", f"${predicted_price:.2f}", f"{price_change:+.2f}%")
        with cols[2]:
            st.metric("Sentiment", f"{sentiment_score:.3f}")
        if is_enhanced and confidence:
            with cols[3]:
                st.metric("Confidence", f"{confidence*100:.1f}%")
        
        # Recommendation
        st.success(f"**Recommendation: {recommendation}**")
    
    def _render_disclaimer(self):
        """Render disclaimer"""
        st.markdown("---")
        st.warning("""
        **Disclaimer:** Educational purposes only. Not financial advice. 
        Always conduct your own research and consult with financial advisors.
        """)
    
    def _show_fallback_interface(self):
        """Show fallback interface when main content fails"""
        st.error("🚨 Application encountered an error")
        st.info("""
        **Troubleshooting steps:**
        1. Click 'Restart Session' in sidebar
        2. Try a different stock symbol
        3. Disable advanced features
        4. Clear browser cache
        """)
        
        if st.button("🔄 Try Basic Analysis"):
            self._restart_session()
    
    def _restart_session(self):
        """Restart the session"""
        for key in list(st.session_state.keys()):
            del st.session_state[key]
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
    
    def _handle_critical_error(self, error):
        """Handle critical application errors"""
        st.error("🚨 Critical Application Error")
        st.code(f"Error details: {str(error)}")
        st.info("""
        **Please try:**
        1. Refreshing the page
        2. Checking your internet connection
        3. Trying a different browser
        4. Contacting support if issue persists
        """)

# Run the application
if __name__ == "__main__":
    app = DeploymentReadyApp()
    app.run()
