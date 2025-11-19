import yfinance as yf
import pandas as pd
import numpy as np
from alpha_vantage.timeseries import TimeSeries
import requests
import os
from datetime import datetime, timedelta
import streamlit as st
import time

class DataFetcher:
    def __init__(self):
        self.alpha_vantage_key = os.getenv('ALPHA_VANTAGE_API_KEY') or st.secrets.get('ALPHA_VANTAGE_API_KEY', 'demo')
        self.gnews_api_key = os.getenv('GNEWS_API_KEY') or st.secrets.get('GNEWS_API_KEY', '')
        
        if self.alpha_vantage_key and self.alpha_vantage_key != 'demo':
            st.sidebar.success("✓ Alpha Vantage API configured!")
        else:
            st.sidebar.warning("Using Alpha Vantage demo key (limited requests)")
            
        if self.gnews_api_key:
            st.sidebar.success("✓ GNews API configured!")
        else:
            st.sidebar.warning("GNews API key not found")
    
    def get_stock_data(self, symbol, period='3mo'):
        """Fetch stock data with multiple fallback sources and better error handling"""
        # Clean symbol
        symbol = symbol.upper().strip()
        
        st.info(f"📊 Fetching data for {symbol}...")
        
        # Try Yahoo Finance first
        data = self._get_stock_data_yahoo_robust(symbol, period)
        if data is not None and not data.empty:
            st.success(f"✓ Yahoo Finance data fetched for {symbol}")
            data = self.calculate_technical_indicators(data)
            return data
        
        # If Yahoo fails, try Alpha Vantage (only if we have a real key)
        if self.alpha_vantage_key and self.alpha_vantage_key != 'demo':
            st.warning("Yahoo Finance failed, trying Alpha Vantage...")
            data = self._get_stock_data_alpha_vantage(symbol)
            if data is not None and not data.empty:
                st.success(f"✓ Alpha Vantage data fetched for {symbol}")
                data = self.calculate_technical_indicators(data)
                return data
        
        # Final fallback - use sample data for demonstration
        st.warning("Using sample data for demonstration...")
        data = self._get_sample_stock_data(symbol)
        if data is not None:
            st.info(f"📋 Showing sample data for {symbol}")
            data = self.calculate_technical_indicators(data)
            return data
        
        st.error(f"❌ Could not fetch data for {symbol}. Try popular symbols like: TSLA, AAPL, GOOGL, MSFT")
        return None
    
    def _get_stock_data_yahoo_robust(self, symbol, period='3mo'):
        """Robust Yahoo Finance data fetching with multiple fallbacks"""
        try:
            # Try direct symbol first
            stock = yf.Ticker(symbol)
            
            # Add small delay to avoid rate limiting
            time.sleep(1)
            
            # Try multiple period formats
            try:
                data = stock.history(period=period)
            except:
                # If period fails, try using start/end dates
                end_date = datetime.now()
                if period == '1mo':
                    start_date = end_date - timedelta(days=30)
                elif period == '3mo':
                    start_date = end_date - timedelta(days=90)
                else:  # 6mo
                    start_date = end_date - timedelta(days=180)
                
                data = stock.history(start=start_date, end=end_date)
            
            if data.empty or len(data) < 5:
                # Try with .NS suffix for Indian stocks
                if not symbol.endswith('.NS'):
                    return self._get_stock_data_yahoo_robust(symbol + '.NS', period)
                return None
            
            # Validate we have required data
            required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            if not all(col in data.columns for col in required_cols):
                return None
            
            return data
            
        except Exception as e:
            st.warning(f"Yahoo Finance temporary error for {symbol}")
            return None
    
    def _get_stock_data_alpha_vantage(self, symbol):
        """Fetch stock data from Alpha Vantage with error handling"""
        try:
            if self.alpha_vantage_key == 'demo':
                st.warning("Alpha Vantage demo key - very limited")
                return None
            
            ts = TimeSeries(key=self.alpha_vantage_key, output_format='pandas')
            
            # Add delay to respect rate limits
            time.sleep(2)
            
            data, meta_data = ts.get_daily(symbol=symbol, outputsize='compact')
            
            if data.empty:
                return None
            
            # Rename columns to match our format
            data.columns = [col.split(' ')[1].capitalize() for col in data.columns]
            
            return data
            
        except Exception as e:
            error_msg = str(e)
            if 'rate limit' in error_msg.lower():
                st.error("⏳ Alpha Vantage rate limit reached. Please try again in a minute.")
            else:
                st.warning(f"Alpha Vantage error: {error_msg}")
            return None
    
    def _get_sample_stock_data(self, symbol):
        """Generate realistic sample stock data for demonstration"""
        try:
            # Create date range for last 90 days
            end_date = datetime.now()
            start_date = end_date - timedelta(days=90)
            dates = pd.date_range(start=start_date, end=end_date, freq='D')
            
            # Remove weekends
            dates = dates[dates.dayofweek < 5]
            
            # Generate realistic price data
            np.random.seed(hash(symbol) % 10000)  # Consistent for same symbol
            
            base_price = 150 + (hash(symbol) % 100)  # Different base price per symbol
            prices = []
            current_price = base_price
            
            for _ in range(len(dates)):
                # Random walk with some trend
                change = np.random.normal(0, 2)
                current_price = max(10, current_price + change)
                prices.append(current_price)
            
            # Create DataFrame
            data = pd.DataFrame({
                'Open': [p * 0.99 for p in prices],
                'High': [p * 1.02 for p in prices],
                'Low': [p * 0.98 for p in prices],
                'Close': prices,
                'Volume': [np.random.randint(1000000, 5000000) for _ in prices]
            }, index=dates)
            
            return data
            
        except Exception as e:
            st.error(f"Error generating sample data: {e}")
            return None
    
    def calculate_technical_indicators(self, df):
        """Calculate technical indicators using pure Python"""
        try:
            # Always available indicators
            df['sma_20'] = df['Close'].rolling(window=20).mean()
            df['sma_50'] = df['Close'].rolling(window=50).mean()
            df['volume_sma'] = df['Volume'].rolling(window=20).mean()
            
            # Try advanced indicators (might fail if insufficient data)
            try:
                # RSI
                df['rsi'] = self._calculate_rsi(df['Close'])
                
                # MACD
                macd, macd_signal, macd_hist = self._calculate_macd(df['Close'])
                df['macd'] = macd
                df['macd_signal'] = macd_signal
                df['macd_hist'] = macd_hist
                
            except Exception as e:
                st.warning(f"Some technical indicators unavailable: {e}")
            
            return df
            
        except Exception as e:
            st.error(f"Error calculating indicators: {e}")
            return df
    
    def _calculate_rsi(self, prices, window=14):
        """Calculate RSI"""
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi
        except:
            return pd.Series([np.nan] * len(prices), index=prices.index)
    
    def _calculate_macd(self, prices, fast=12, slow=26, signal=9):
        """Calculate MACD"""
        try:
            ema_fast = prices.ewm(span=fast).mean()
            ema_slow = prices.ewm(span=slow).mean()
            macd = ema_fast - ema_slow
            macd_signal = macd.ewm(span=signal).mean()
            macd_hist = macd - macd_signal
            return macd, macd_signal, macd_hist
        except:
            nan_series = pd.Series([np.nan] * len(prices), index=prices.index)
            return nan_series, nan_series, nan_series
    
    def get_news_data(self, symbol, days=7):
        """Fetch news data with better error handling"""
        # Try GNews API first
        if self.gnews_api_key:
            gnews_data = self._get_news_gnews(symbol)
            if not gnews_data.empty:
                return gnews_data
        
        # Fallback to sample news data
        return self._get_sample_news_data(symbol)
    
    def _get_news_gnews(self, symbol):
        """Fetch news using GNews API"""
        try:
            url = f"https://gnews.io/api/v4/search?q={symbol} stock&token={self.gnews_api_key}&lang=en&max=5"
            
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                articles = data.get('articles', [])
                
                if articles:
                    news_data = []
                    for article in articles:
                        news_data.append({
                            'title': article['title'],
                            'description': article.get('description', ''),
                            'published_at': article['publishedAt'],
                            'source': article['source']['name']
                        })
                    
                    return pd.DataFrame(news_data)
        except Exception as e:
            st.warning(f"GNews API error: {e}")
        
        return pd.DataFrame()
    
    def _get_sample_news_data(self, symbol):
        """Generate sample news data"""
        sample_news = [
            {
                'title': f'Market analysis for {symbol} shows positive trends',
                'description': f'Technical indicators suggest favorable conditions for {symbol} in current market.',
                'published_at': datetime.now().isoformat(),
                'source': 'Market Analysis'
            },
            {
                'title': f'{symbol} demonstrates strong fundamentals',
                'description': f'Recent performance indicators show {symbol} maintaining competitive position.',
                'published_at': (datetime.now() - timedelta(hours=2)).isoformat(),
                'source': 'Financial Review'
            }
        ]
        return pd.DataFrame(sample_news)
