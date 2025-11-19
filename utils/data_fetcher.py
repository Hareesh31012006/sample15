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
                
                # Bollinger Bands
                bb_upper, bb_middle, bb_lower = self._calculate_bollinger_bands(df['Close'])
                df['bb_upper'] = bb_upper
                df['bb_middle'] = bb_middle
                df['bb_lower'] = bb_lower
                
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
    
    def _calculate_bollinger_bands(self, prices, window=20, num_std=2):
        """Calculate Bollinger Bands"""
        try:
            middle_band = prices.rolling(window=window).mean()
            std = prices.rolling(window=window).std()
            upper_band = middle_band + (std * num_std)
            lower_band = middle_band - (std * num_std)
            return upper_band, middle_band, lower_band
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
        """Generate more realistic sample news data"""
        current_time = datetime.now()
        
        sample_news = [
            {
                'title': f'{symbol} Shows Strong Quarterly Earnings Growth',
                'description': f'{symbol} reported better-than-expected earnings this quarter, with revenue growth exceeding analyst projections by 15%. The company continues to demonstrate robust financial performance.',
                'published_at': current_time.strftime('%Y-%m-%dT%H:%M:%SZ'),
                'source': 'Financial Times'
            },
            {
                'title': f'Analysts Upgrade {symbol} Stock Rating to "Buy"',
                'description': f'Several major investment firms have upgraded their rating on {symbol} citing strong market position and growth potential in emerging markets.',
                'published_at': (current_time - timedelta(hours=2)).strftime('%Y-%m-%dT%H:%M:%SZ'),
                'source': 'Market Watch'
            },
            {
                'title': f'{symbol} Announces Revolutionary New Product Launch',
                'description': f'The company unveiled its latest product line, expected to drive significant revenue growth in the coming quarters and capture new market segments.',
                'published_at': (current_time - timedelta(days=1)).strftime('%Y-%m-%dT%H:%M:%SZ'),
                'source': 'Tech News Daily'
            },
            {
                'title': f'{symbol} Faces Regulatory Challenges in European Market',
                'description': f'Recent regulatory changes may impact {symbol} operations in key European markets, according to industry analysts. The company is working closely with regulators.',
                'published_at': (current_time - timedelta(days=2)).strftime('%Y-%m-%dT%H:%M:%SZ'),
                'source': 'Business Daily'
            },
            {
                'title': f'Institutional Investors Increase Stakes in {symbol}',
                'description': f'Recent SEC filings show major institutional investors have significantly increased their positions in {symbol}, signaling strong confidence in long-term growth prospects.',
                'published_at': (current_time - timedelta(days=3)).strftime('%Y-%m-%dT%H:%M:%SZ'),
                'source': 'Investor Digest'
            },
            {
                'title': f'{symbol} Expands into Asian Markets with New Partnerships',
                'description': f'The company announced strategic partnerships in key Asian markets, positioning itself for international expansion and diversified revenue streams.',
                'published_at': (current_time - timedelta(days=4)).strftime('%Y-%m-%dT%H:%M:%SZ'),
                'source': 'Global Business Review'
            },
            {
                'title': f'{symbol} CEO Discusses Future Innovation Strategy',
                'description': f'In a recent interview, the CEO outlined the company\'s commitment to innovation and R&D investment, highlighting upcoming technological advancements.',
                'published_at': (current_time - timedelta(days=5)).strftime('%Y-%m-%dT%H:%M:%SZ'),
                'source': 'Executive Insights'
            }
        ]
        return pd.DataFrame(sample_news)
    
    def get_market_summary(self, symbols=['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN']):
        """Get quick market summary for major stocks"""
        try:
            summary_data = []
            for symbol in symbols[:3]:  # Limit to 3 for performance
                try:
                    data = self.get_stock_data(symbol, '1d')
                    if data is not None and not data.empty:
                        current_price = data['Close'].iloc[-1]
                        prev_price = data['Close'].iloc[-2] if len(data) > 1 else current_price
                        change = current_price - prev_price
                        change_percent = (change / prev_price) * 100
                        
                        summary_data.append({
                            'Symbol': symbol,
                            'Price': f"${current_price:.2f}",
                            'Change': f"{change:+.2f}",
                            'Change %': f"{change_percent:+.2f}%",
                            'Trend': '📈' if change > 0 else '📉' if change < 0 else '➡️'
                        })
                except:
                    continue
            
            return pd.DataFrame(summary_data)
        except Exception as e:
            st.warning(f"Market summary error: {e}")
            return pd.DataFrame()
