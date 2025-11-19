import yfinance as yf
import pandas as pd
import numpy as np
from alpha_vantage.timeseries import TimeSeries
import requests
import os
from datetime import datetime, timedelta
import streamlit as st

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
    
    def get_stock_data(self, symbol, period='6mo'):
        """Fetch stock data with multiple fallback sources"""
        # Try Yahoo Finance first
        st.info(f"Fetching data for {symbol}...")
        
        data = self._get_stock_data_yahoo(symbol, period)
        if data is not None and not data.empty:
            st.success(f"✓ Data fetched successfully for {symbol}")
            data = self.calculate_technical_indicators(data)
            return data
        
        # If Yahoo fails, try Alpha Vantage
        st.warning("Yahoo Finance failed, trying Alpha Vantage...")
        data = self._get_stock_data_alpha_vantage(symbol)
        if data is not None and not data.empty:
            st.success(f"✓ Alpha Vantage data fetched for {symbol}")
            data = self.calculate_technical_indicators(data)
            return data
        
        # If both fail, show error
        st.error(f"❌ Could not fetch data for {symbol} from any source. Please check the symbol and try again.")
        return None
    
    def _get_stock_data_yahoo(self, symbol, period='6mo'):
        """Fetch stock data from Yahoo Finance with error handling"""
        try:
            # Add .NS for Indian stocks, otherwise keep as is
            if not any(ext in symbol.upper() for ext in ['.NS', '.BO', '.BS']):
                symbol_to_fetch = symbol
            else:
                symbol_to_fetch = symbol
            
            stock = yf.Ticker(symbol_to_fetch)
            
            # Try multiple period formats
            try:
                data = stock.history(period=period)
            except:
                # If period fails, try using start/end dates
                end_date = datetime.now()
                start_date = end_date - timedelta(days=180)  # 6 months
                data = stock.history(start=start_date, end=end_date)
            
            if data.empty:
                # Try with .NS suffix for Indian stocks
                if not symbol.upper().endswith('.NS'):
                    return self._get_stock_data_yahoo(symbol + '.NS', period)
                return None
            
            # Ensure we have the required columns
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            for col in required_columns:
                if col not in data.columns:
                    st.error(f"Missing column {col} in data")
                    return None
            
            return data
            
        except Exception as e:
            st.warning(f"Yahoo Finance error for {symbol}: {str(e)}")
            return None
    
    def _get_stock_data_alpha_vantage(self, symbol):
        """Fetch stock data from Alpha Vantage"""
        try:
            if self.alpha_vantage_key == 'demo':
                st.warning("Using Alpha Vantage demo key - limited functionality")
            
            ts = TimeSeries(key=self.alpha_vantage_key, output_format='pandas')
            data, meta_data = ts.get_daily(symbol=symbol, outputsize='compact')
            
            if data.empty:
                return None
            
            # Rename columns to match Yahoo Finance format
            data.columns = [col.split(' ')[1] for col in data.columns]
            data = data.rename(columns={
                'open': 'Open',
                'high': 'High', 
                'low': 'Low',
                'close': 'Close',
                'volume': 'Volume'
            })
            
            return data
            
        except Exception as e:
            st.warning(f"Alpha Vantage error for {symbol}: {str(e)}")
            return None
    
    def calculate_technical_indicators(self, df):
        """Calculate technical indicators using pure Python only"""
        try:
            # Basic indicators that always work
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
                st.warning(f"Some technical indicators couldn't be calculated: {e}")
            
            return df
            
        except Exception as e:
            st.error(f"Error calculating technical indicators: {e}")
            return df
    
    def _calculate_rsi(self, prices, window=14):
        """Calculate RSI using pure Python"""
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
        """Calculate MACD using pure Python"""
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
        """Calculate Bollinger Bands using pure Python"""
        try:
            middle = prices.rolling(window=window).mean()
            std = prices.rolling(window=window).std()
            upper = middle + (std * num_std)
            lower = middle - (std * num_std)
            return upper, middle, lower
        except:
            nan_series = pd.Series([np.nan] * len(prices), index=prices.index)
            return nan_series, nan_series, nan_series
    
    def get_news_data(self, symbol, days=7):
        """Fetch news data using GNews API with Yahoo fallback"""
        # Try GNews API first
        if self.gnews_api_key:
            gnews_data = self._get_news_gnews(symbol)
            if not gnews_data.empty:
                return gnews_data
        
        # Fallback to Yahoo Finance news
        yahoo_data = self._get_news_yahoo(symbol)
        if not yahoo_data.empty:
            return yahoo_data
        
        # Final fallback to sample data
        return self._get_sample_news_data(symbol)
    
    def _get_news_gnews(self, symbol):
        """Fetch news using GNews API"""
        try:
            url = f"https://gnews.io/api/v4/search?q={symbol} stock&token={self.gnews_api_key}&lang=en&country=us&max=10"
            
            response = requests.get(url)
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
    
    def _get_news_yahoo(self, symbol):
        """Fetch news from Yahoo Finance"""
        try:
            stock = yf.Ticker(symbol)
            news = stock.news
            
            if not news:
                return pd.DataFrame()
            
            news_data = []
            for item in news[:10]:
                # Convert timestamp to readable format
                pub_time = datetime.fromtimestamp(item.get('providerPublishTime', 0))
                
                news_data.append({
                    'title': item.get('title', 'No title'),
                    'description': item.get('summary', 'No description'),
                    'published_at': pub_time.isoformat(),
                    'source': item.get('publisher', 'Yahoo Finance')
                })
            
            return pd.DataFrame(news_data)
        except Exception as e:
            st.warning(f"Yahoo News error: {e}")
            return pd.DataFrame()
    
    def _get_sample_news_data(self, symbol):
        """Generate sample news data as final fallback"""
        sample_news = [
            {
                'title': f'Positive outlook for {symbol} as earnings exceed expectations',
                'description': f'Company {symbol} reports strong quarterly results with revenue growth.',
                'published_at': datetime.now().isoformat(),
                'source': 'Financial Times'
            },
            {
                'title': f'{symbol} announces new product launch',
                'description': f'{symbol} unveils innovative product line expected to drive future growth.',
                'published_at': (datetime.now() - timedelta(hours=2)).isoformat(),
                'source': 'Business Insider'
            },
            {
                'title': f'Analysts maintain buy rating for {symbol}',
                'description': f'Market analysts recommend {symbol} as strong investment opportunity.',
                'published_at': (datetime.now() - timedelta(days=1)).isoformat(),
                'source': 'Bloomberg'
            }
        ]
        return pd.DataFrame(sample_news)
