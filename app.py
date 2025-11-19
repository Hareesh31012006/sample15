import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os
import gc
import psutil
import warnings
warnings.filterwarnings('ignore')

# Configure Streamlit page
st.set_page_config(
    page_title="NextTick - AI Stock Prediction",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import modules with memory optimization
try:
    from utils.data_fetcher import DataFetcher
    from utils.sentiment_analyzer import OptimizedSentimentAnalyzer
    from utils.model import StockPredictor
    from utils.visualization import StockVisualizer
except ImportError as e:
    st.error(f"❌ Module import error: {e}")
    st.stop()

class MemoryOptimizedStockApp:
    def __init__(self):
        self.data_fetcher = None
        self.sentiment_analyzer = None
        self.stock_predictor = None
        self.visualizer = None
        self.analysis_count = 0
        self._initialize_components()
        self._initialize_session_state()
    
    def _initialize_components(self):
        """Initialize components with memory optimization"""
        try:
            self.data_fetcher = DataFetcher()
            self.sentiment_analyzer = OptimizedSentimentAnalyzer()
            self.stock_predictor = StockPredictor()
            self.visualizer = StockVisualizer()
        except Exception as e:
            st.error(f"❌ Component initialization failed: {e}")
            st.stop()
    
    def _initialize_session_state(self):
        """Initialize all required session state variables"""
        if 'current_symbol' not in st.session_state:
            st.session_state.current_symbol = 'AAPL'
        if 'period' not in st.session_state:
            st.session_state.period = '3mo'
        if 'use_technical_indicators' not in st.session_state:
            st.session_state.use_technical_indicators = True
        if 'use_sentiment_analysis' not in st.session_state:
            st.session_state.use_sentiment_analysis = True
        if 'use_advanced_model' not in st.session_state:
            st.session_state.use_advanced_model = True
        if 'analysis_count' not in st.session_state:
            st.session_state.analysis_count = 0
        if 'current_cache_key' not in st.session_state:
            st.session_state.current_cache_key = ''
    
    def _get_memory_usage(self):
        """Get current memory usage in MB"""
        try:
            process = psutil.Process(os.getpid())
            return process.memory_info().rss / 1024 / 1024
        except:
            return 0
    
    def _check_memory_health(self):
        """Check if memory usage is getting too high"""
        current_memory = self._get_memory_usage()
        
        # Show memory usage in sidebar
        if current_memory > 0:
            memory_color = "🟢" if current_memory < 500 else "🟡" if current_memory < 800 else "🔴"
            st.sidebar.write(f"{memory_color} Memory: {current_memory:.0f}MB")
        
        # Auto-cleanup if memory gets too high
        if current_memory > 800:  # 800MB threshold
            st.sidebar.warning("🔄 High memory usage - auto-cleaning...")
            self._force_memory_cleanup()
            return True
        return False
    
    def _force_memory_cleanup(self):
        """Forceful memory cleanup to prevent crashes"""
        try:
            # Clean up sentiment analyzer (biggest memory user)
            if hasattr(self, 'sentiment_analyzer') and self.sentiment_analyzer:
                self.sentiment_analyzer.cleanup()
            
            # Clean up stock predictor
            if hasattr(self, 'stock_predictor') and self.stock_predictor:
                self.stock_predictor.cleanup()
            
            # Clear session state data except essential variables
            essential_keys = ['current_symbol', 'period', 'use_technical_indicators', 
                            'use_sentiment_analysis', 'use_advanced_model', 'analysis_count']
            keys_to_delete = [key for key in st.session_state.keys() if key not in essential_keys]
            
            for key in keys_to_delete:
                if key in st.session_state:
                    del st.session_state[key]
            
            # Force garbage collection
            gc.collect()
            
            # Reinitialize components
            self._initialize_components()
            
            st.sidebar.success("🧹 Memory cleaned!")
            
        except Exception as e:
            st.sidebar.warning(f"Memory cleanup note: {e}")
    
    def _smart_analysis_management(self):
        """Manage analysis count and auto-cleanup"""
        if 'analysis_count' not in st.session_state:
            st.session_state.analysis_count = 0
        
        st.session_state.analysis_count += 1
        self.analysis_count = st.session_state.analysis_count
        
        # Auto-cleanup every analysis to prevent memory buildup
        if st.session_state.analysis_count >= 1:  # Clean after every analysis
            self._gentle_memory_cleanup()
        
        # Always check memory health
        memory_issue = self._check_memory_health()
        return memory_issue
    
    def _gentle_memory_cleanup(self):
        """Gentle cleanup between analyses"""
        try:
            # Clean up sentiment analyzer but keep model loaded
            if hasattr(self, 'sentiment_analyzer') and self.sentiment_analyzer:
                self.sentiment_analyzer.cleanup()
            
            # Clean up stock predictor
            if hasattr(self, 'stock_predictor') and self.stock_predictor:
                self.stock_predictor.cleanup()
            
            # Clear large data objects
            if 'stock_data' in st.session_state:
                del st.session_state.stock_data
            if 'news_data' in st.session_state:
                del st.session_state.news_data
            if 'article_sentiments' in st.session_state:
                del st.session_state.article_sentiments
            
            gc.collect()
            
        except Exception as e:
            # Silent cleanup - don't show errors to user
            pass
    
    def run(self):
        """Main application runner"""
        # Custom CSS
        st.markdown("""
        <style>
        .main-header {
            font-size: 2.8rem;
            color: #1f77b4;
            text-align: center;
            margin-bottom: 2rem;
            font-weight: bold;
        }
        .memory-optimized {
            background: linear-gradient(45deg, #667eea, #764ba2);
            color: white;
            padding: 6px 12px;
            border-radius: 20px;
            font-size: 0.9em;
            font-weight: bold;
        }
        .news-article {
            background: #f8f9fa;
            padding: 1rem;
            margin: 0.5rem 0;
            border-radius: 8px;
            border-left: 4px solid #1f77b4;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Header
        st.markdown(
            '<h1 class="main-header">🚀 NextTick AI <span class="memory-optimized">Memory Optimized</span></h1>', 
            unsafe_allow_html=True
        )
        
        # Sidebar
        self._render_sidebar()
        
        # Main analysis
        symbol = st.session_state.get('current_symbol', 'AAPL')
        if symbol:
            self.analyze_stock(symbol)
    
    def _render_sidebar(self):
        """Render sidebar configuration"""
        st.sidebar.title("⚙️ Configuration")
        
        # Stock symbol input
        symbol = st.sidebar.text_input("Enter Stock Symbol", st.session_state.current_symbol).upper()
        st.session_state.current_symbol = symbol
        
        # Analysis period
        period = st.sidebar.selectbox("Data Period", ["1mo", "3mo", "6mo"], 
                                    index=["1mo", "3mo", "6mo"].index(st.session_state.period))
        st.session_state.period = period
        
        # Features
        st.sidebar.subheader("🎯 Features")
        use_technical_indicators = st.sidebar.checkbox("Technical Indicators", 
                                                     st.session_state.use_technical_indicators)
        use_sentiment_analysis = st.sidebar.checkbox("Sentiment Analysis", 
                                                   st.session_state.use_sentiment_analysis)
        use_advanced_model = st.sidebar.checkbox("Enhanced AI Model", 
                                               st.session_state.use_advanced_model)
        
        st.session_state.use_technical_indicators = use_technical_indicators
        st.session_state.use_sentiment_analysis = use_sentiment_analysis
        st.session_state.use_advanced_model = use_advanced_model
        
        # Memory management
        st.sidebar.markdown("---")
        st.sidebar.subheader("🧠 Memory Management")
        
        if st.sidebar.button("🔄 Clear Memory", help="Force clear memory to prevent crashes"):
            self._force_memory_cleanup()
            st.rerun()
        
        if st.sidebar.button("🆕 New Analysis", help="Start fresh analysis"):
            self._force_memory_cleanup()
            st.session_state.analysis_count = 0
            st.rerun()
        
        # Analysis counter
        if 'analysis_count' in st.session_state:
            st.sidebar.write(f"📊 Analyses: {st.session_state.analysis_count}")
    
    def analyze_stock(self, symbol):
        """Perform memory-optimized stock analysis"""
        try:
            # Apply smart memory management
            memory_issue = self._smart_analysis_management()
            if memory_issue:
                st.info("🔄 Memory optimized for better performance")
            
            # Fetch data
            stock_data = self._fetch_stock_data(symbol, st.session_state.period)
            if stock_data is None:
                return
            
            # Fetch news if needed
            news_data = pd.DataFrame()
            if st.session_state.use_sentiment_analysis:
                news_data = self._fetch_news_data(symbol)
            
            # Display analysis
            self._display_stock_overview(stock_data, symbol)
            self._display_price_chart(stock_data, symbol)
            
            if st.session_state.use_technical_indicators:
                self._display_technical_analysis(stock_data)
            
            sentiment_score = 0.0
            article_sentiments = []
            if st.session_state.use_sentiment_analysis and not news_data.empty:
                sentiment_score, article_sentiments = self._display_sentiment_analysis(news_data, symbol)
            
            self._display_ai_prediction(stock_data, sentiment_score, symbol)
            
        except Exception as e:
            st.error(f"❌ Analysis error: {str(e)}")
            # Auto-recover on error
            self._force_memory_cleanup()
    
    def _fetch_stock_data(self, symbol, period):
        """Fetch stock data with caching"""
        cache_key = f"{symbol}_{period}"
        
        if 'stock_data' not in st.session_state or st.session_state.get('current_cache_key') != cache_key:
            with st.spinner(f"📊 Fetching data for {symbol}..."):
                stock_data = self.data_fetcher.get_stock_data(symbol, period)
                
                if stock_data is None or stock_data.empty:
                    st.error(f"❌ No data found for {symbol}")
                    return None
                
                st.session_state.stock_data = stock_data
                st.session_state.current_cache_key = cache_key
        
        return st.session_state.stock_data
    
    def _fetch_news_data(self, symbol):
        """Fetch news data with error handling"""
        try:
            with st.spinner("📰 Fetching news..."):
                return self.data_fetcher.get_news_data(symbol)
        except Exception as e:
            st.warning(f"⚠️ News fetching failed: {e}")
            return pd.DataFrame()
    
    def _display_stock_overview(self, stock_data, symbol):
        """Display stock overview"""
        st.subheader(f"📈 {symbol} Overview")
        
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
            if 'rsi' in stock_data.columns and not pd.isna(stock_data['rsi'].iloc[-1]):
                rsi = stock_data['rsi'].iloc[-1]
                rsi_color = "🟢" if rsi < 30 else "🔴" if rsi > 70 else "🟡"
                st.metric("RSI", f"{rsi:.1f}", delta=None, delta_color="off")
                st.write(f"{rsi_color} {'Oversold' if rsi < 30 else 'Overbought' if rsi > 70 else 'Neutral'}")
    
    def _display_price_chart(self, stock_data, symbol):
        """Display price chart"""
        st.subheader("📊 Price Chart")
        try:
            price_chart = self.visualizer.plot_stock_price(stock_data, f"{symbol} Price")
            st.plotly_chart(price_chart, use_container_width=True)
        except Exception as e:
            st.error(f"❌ Chart error: {e}")
    
    def _display_technical_analysis(self, stock_data):
        """Display technical analysis"""
        st.subheader("🔧 Technical Analysis")
        try:
            tech_fig = self.visualizer.plot_technical_indicators(stock_data)
            st.pyplot(tech_fig)
        except Exception as e:
            st.warning(f"⚠️ Technical analysis error: {e}")
    
    def _display_sentiment_analysis(self, news_data, symbol):
        """Display sentiment analysis with news articles"""
        st.subheader("😊 Sentiment Analysis")
        
        try:
            with st.spinner("Analyzing sentiment..."):
                sentiment_score, article_sentiments = self.sentiment_analyzer.analyze_news_sentiment_detailed(news_data)
            
            # Display sentiment gauge
            sentiment_chart = self.visualizer.plot_sentiment_analysis(sentiment_score, news_data)
            st.plotly_chart(sentiment_chart, use_container_width=True)
            
            # Interpret sentiment
            self._interpret_sentiment(sentiment_score)
            
            # Display analyzed news articles
            self._display_news_articles(article_sentiments, symbol)
            
            return sentiment_score, article_sentiments
            
        except Exception as e:
            st.warning(f"⚠️ Sentiment analysis error: {e}")
            return 0.0, []
    
    def _display_news_articles(self, article_sentiments, symbol):
        """Display analyzed news articles"""
        if not article_sentiments:
            st.info("No news articles available for analysis.")
            return
        
        st.subheader(f"📰 Recent News for {symbol}")
        
        for i, article in enumerate(article_sentiments):
            with st.container():
                st.markdown(f"""
                <div class="news-article">
                    <h4>📖 {article['title']}</h4>
                    <p><strong>Source:</strong> {article.get('source', 'Unknown')} | 
                    <strong>Published:</strong> {article.get('published_at', 'Unknown')}</p>
                    <p>{article.get('description', 'No description available.')}</p>
                    <div style="display: flex; gap: 1rem; margin-top: 0.5rem;">
                        <span style="background: {'#4CAF50' if article['sentiment'] > 0.1 else '#FF9800' if article['sentiment'] < -0.1 else '#9E9E9E'}; 
                        color: white; padding: 0.2rem 0.5rem; border-radius: 4px;">
                            Sentiment: {article['sentiment']:.3f}
                        </span>
                        <span style="background: #2196F3; color: white; padding: 0.2rem 0.5rem; border-radius: 4px;">
                            {article['label']}
                        </span>
                        <span style="background: #673AB7; color: white; padding: 0.2rem 0.5rem; border-radius: 4px;">
                            Confidence: {article['confidence']:.1%}
                        </span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
    
    def _interpret_sentiment(self, sentiment_score):
        """Interpret sentiment score"""
        if sentiment_score > 0.3:
            interpretation, color = "Strongly Bullish 🚀", "#4CAF50"
        elif sentiment_score > 0.1:
            interpretation, color = "Bullish 📈", "#8BC34A"
        elif sentiment_score > -0.1:
            interpretation, color = "Neutral ➡️", "#9E9E9E"
        elif sentiment_score > -0.3:
            interpretation, color = "Bearish 📉", "#FF9800"
        else:
            interpretation, color = "Strongly Bearish 🔻", "#F44336"
        
        st.info(f"**Overall Market Sentiment:** {interpretation} (Score: {sentiment_score:.3f})")
    
    def _display_ai_prediction(self, stock_data, sentiment_score, symbol):
        """Display AI prediction"""
        st.subheader("🤖 AI Prediction")
        
        try:
            if st.session_state.use_advanced_model:
                self._run_enhanced_prediction(stock_data, sentiment_score, symbol)
            else:
                self._run_basic_prediction(stock_data, sentiment_score, symbol)
                
        except Exception as e:
            st.error(f"❌ Prediction error: {e}")
    
    def _run_enhanced_prediction(self, stock_data, sentiment_score, symbol):
        """Run enhanced prediction"""
        with st.spinner("🧠 Training advanced model..."):
            feature_columns = ['Close', 'Volume']
            if 'rsi' in stock_data.columns:
                feature_columns.append('rsi')
            if 'macd' in stock_data.columns:
                feature_columns.append('macd')
            
            train_data = stock_data[feature_columns].dropna()
            
            if len(train_data) > 10:
                try:
                    success = self.stock_predictor.train_enhanced_model(train_data, feature_columns)
                    
                    if success:
                        predicted_price, current_price, confidence = self.stock_predictor.predict_next_day(train_data, feature_columns)
                        
                        if predicted_price is not None:
                            self._display_prediction_results(
                                predicted_price, current_price, sentiment_score, confidence, True
                            )
                        else:
                            st.warning("❌ Enhanced prediction failed")
                    else:
                        st.warning("❌ Enhanced model training failed")
                        
                except Exception as e:
                    st.warning(f"Enhanced model error: {e}")
            else:
                st.warning("❌ Insufficient data for enhanced model")
    
    def _run_basic_prediction(self, stock_data, sentiment_score, symbol):
        """Run basic prediction"""
        with st.spinner("🧠 Training model..."):
            feature_columns = ['Close', 'Volume']
            train_data = stock_data[feature_columns].dropna()
            
            if len(train_data) > 5:
                try:
                    # Use simple linear regression for memory efficiency
                    success = self.stock_predictor.train_linear_regression(train_data, feature_columns)
                    
                    if success:
                        predicted_price, current_price, confidence = self.stock_predictor.predict_next_day_simple(train_data, feature_columns)
                        
                        if predicted_price is not None:
                            self._display_prediction_results(
                                predicted_price, current_price, sentiment_score, confidence, False
                            )
                        else:
                            st.error("❌ Prediction generation failed")
                    else:
                        st.error("❌ Model training failed")
                        
                except Exception as e:
                    st.error(f"❌ Prediction error: {e}")
            else:
                st.error("❌ Insufficient data for prediction")
    
    def _display_prediction_results(self, predicted_price, current_price, sentiment_score, confidence, is_enhanced):
        """Display prediction results"""
        # Calculate price change
        price_change = ((predicted_price - current_price) / current_price) * 100
        
        # Generate recommendation
        recommendation, reasoning, color, emoji = self._generate_recommendation(
            price_change, sentiment_score, confidence
        )
        
        # Display metrics
        if is_enhanced and confidence:
            cols = st.columns(4)
            with cols[0]:
                st.metric("Current Price", f"${current_price:.2f}")
            with cols[1]:
                st.metric("Predicted Price", f"${predicted_price:.2f}", f"{price_change:+.2f}%")
            with cols[2]:
                st.metric("Sentiment", f"{sentiment_score:.3f}")
            with cols[3]:
                st.metric("Confidence", f"{confidence*100:.1f}%")
        else:
            cols = st.columns(3)
            with cols[0]:
                st.metric("Current Price", f"${current_price:.2f}")
            with cols[1]:
                st.metric("Predicted Price", f"${predicted_price:.2f}", f"{price_change:+.2f}%")
            with cols[2]:
                st.metric("Sentiment", f"{sentiment_score:.3f}")
        
        # Display recommendation
        st.markdown(f"""
        <div style='background-color: {color}20; padding: 1.5rem; border-radius: 10px; border-left: 5px solid {color}; margin: 1rem 0;'>
            <h2 style='color: {color}; margin: 0;'>{emoji} {recommendation}</h2>
            <p style='margin: 0.5rem 0 0 0; color: #666;'><strong>Reasoning:</strong> {reasoning}</p>
        </div>
        """, unsafe_allow_html=True)
    
    def _generate_recommendation(self, price_change, sentiment_score, confidence):
        """Generate trading recommendation"""
        # Strong signals
        if price_change > 5.0 and sentiment_score > 0.4:
            return "STRONG BUY", "Exceptional growth potential with very positive sentiment", "#4CAF50", "🚀"
        elif price_change > 3.0 and sentiment_score > 0.2:
            return "BUY", "Strong growth expected with positive sentiment", "#8BC34A", "📈"
        elif price_change > 1.5 and sentiment_score > 0.1:
            return "CAUTIOUS BUY", "Moderate growth with favorable sentiment", "#CDDC39", "↗️"
        elif price_change < -5.0 and sentiment_score < -0.4:
            return "STRONG SELL", "Significant decline with very negative sentiment", "#F44336", "🔻"
        elif price_change < -3.0 and sentiment_score < -0.2:
            return "SELL", "Substantial decline with negative sentiment", "#FF9800", "📉"
        elif price_change < -1.5 and sentiment_score < -0.1:
            return "CAUTIOUS SELL", "Moderate decline with concerning sentiment", "#FFC107", "↘️"
        
        # Mixed signals
        elif price_change > 1.0 and sentiment_score < -0.1:
            if confidence and confidence > 0.7:
                return "CAUTIOUS BUY", "Price growth outweighs negative sentiment", "#CDDC39", "↗️"
            else:
                return "HOLD", "Conflicting signals - price up but sentiment negative", "#9E9E9E", "⚖️"
        elif price_change < -1.0 and sentiment_score > 0.1:
            if confidence and confidence > 0.7:
                return "CAUTIOUS SELL", "Price decline outweighs positive sentiment", "#FFC107", "↘️"
            else:
                return "HOLD", "Conflicting signals - price down but sentiment positive", "#9E9E9E", "⚖️"
        
        # Weak signals
        else:
            return "HOLD", "Insufficient signals for clear direction", "#9E9E9E", "⚖️"

# Run the application
if __name__ == "__main__":
    app = MemoryOptimizedStockApp()
    app.run()
