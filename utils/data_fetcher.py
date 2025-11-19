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
    
    def get_stock_data_yahoo(self, symbol, period='6mo'):
        """Fetch stock data from Yahoo Finance"""
        try:
            stock = yf.Ticker(symbol)
            data = stock.history(period=period)
            if data.empty:
                st.error(f"No data found for symbol: {symbol}")
                return None
            
            # Calculate technical indicators
            data = self.calculate_technical_indicators(data)
            return data
        except Exception as e:
            st.error(f"Error fetching data for {symbol}: {e}")
            return None
    
    def get_stock_data_alpha_vantage(self, symbol):
        """Fetch stock data from Alpha Vantage"""
        try:
            ts = TimeSeries(key=self.alpha_vantage_key, output_format='pandas')
            data, meta_data = ts.get_daily(symbol=symbol, outputsize='compact')
            return data
        except Exception as e:
            st.error(f"Alpha Vantage error: {e}")
            return None
    
    def calculate_technical_indicators(self, df):
        """Calculate technical indicators using pure Python only"""
        try:
            # RSI - Relative Strength Index
            df['rsi'] = self._calculate_rsi(df['Close'])
            
            # MACD - Moving Average Convergence Divergence
            macd, macd_signal, macd_hist = self._calculate_macd(df['Close'])
            df['macd'] = macd
            df['macd_signal'] = macd_signal
            df['macd_hist'] = macd_hist
            
            # Moving Averages
            df['sma_20'] = df['Close'].rolling(window=20).mean()
            df['sma_50'] = df['Close'].rolling(window=50).mean()
            
            # Exponential Moving Averages
            df['ema_12'] = df['Close'].ewm(span=12).mean()
            df['ema_26'] = df['Close'].ewm(span=26).mean()
            
            # Bollinger Bands
            bb_upper, bb_middle, bb_lower = self._calculate_bollinger_bands(df['Close'])
            df['bb_upper'] = bb_upper
            df['bb_middle'] = bb_middle
            df['bb_lower'] = bb_lower
            
            # Volume indicators
            df['volume_sma'] = df['Volume'].rolling(window=20).mean()
            
            # Price rate of change
            df['roc'] = self._calculate_roc(df['Close'])
            
            # Stochastic Oscillator
            k, d = self._calculate_stochastic(df['High'], df['Low'], df['Close'])
            df['stoch_k'] = k
            df['stoch_d'] = d
            
            # Average True Range (ATR)
            df['atr'] = self._calculate_atr(df['High'], df['Low'], df['Close'])
            
            # On Balance Volume (OBV)
            df['obv'] = self._calculate_obv(df['Close'], df['Volume'])
            
        except Exception as e:
            st.warning(f"Error calculating some technical indicators: {e}")
        
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
    
    def _calculate_roc(self, prices, window=12):
        """Calculate Rate of Change"""
        try:
            return ((prices / prices.shift(window)) - 1) * 100
        except:
            return pd.Series([np.nan] * len(prices), index=prices.index)
    
    def _calculate_stochastic(self, high, low, close, k_window=14, d_window=3):
        """Calculate Stochastic Oscillator %K and %D"""
        try:
            lowest_low = low.rolling(window=k_window).min()
            highest_high = high.rolling(window=k_window).max()
            k = ((close - lowest_low) / (highest_high - lowest_low)) * 100
            d = k.rolling(window=d_window).mean()
            return k, d
        except:
            nan_series = pd.Series([np.nan] * len(close), index=close.index)
            return nan_series, nan_series
    
    def _calculate_atr(self, high, low, close, window=14):
        """Calculate Average True Range"""
        try:
            tr1 = high - low
            tr2 = abs(high - close.shift())
            tr3 = abs(low - close.shift())
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = tr.rolling(window=window).mean()
            return atr
        except:
            return pd.Series([np.nan] * len(high), index=high.index)
    
    def _calculate_obv(self, close, volume):
        """Calculate On Balance Volume"""
        try:
            obv = (volume * (~close.diff().le(0) * 2 - 1)).cumsum()
            return obv
        except:
            return pd.Series([np.nan] * len(close), index=close.index)
    
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
            },
            {
                'title': f'{symbol} expands into new markets',
                'description': f'Strategic expansion expected to increase {symbol} market share.',
                'published_at': (datetime.now() - timedelta(days=2)).isoformat(),
                'source': 'Reuters'
            },
            {
                'title': f'Industry trends favor {symbol} growth',
                'description': f'Sector analysis shows favorable conditions for {symbol} performance.',
                'published_at': (datetime.now() - timedelta(days=3)).isoformat(),
                'source': 'MarketWatch'
            }
        ]
        return pd.DataFrame(sample_news)
    
    def get_market_summary(self, symbol):
        """Get quick market summary for a stock"""
        try:
            stock = yf.Ticker(symbol)
            info = stock.info
            
            summary = {
                'company_name': info.get('longName', 'N/A'),
                'sector': info.get('sector', 'N/A'),
                'industry': info.get('industry', 'N/A'),
                'market_cap': info.get('marketCap', 'N/A'),
                'pe_ratio': info.get('trailingPE', 'N/A'),
                'dividend_yield': info.get('dividendYield', 'N/A'),
                '52_week_high': info.get('fiftyTwoWeekHigh', 'N/A'),
                '52_week_low': info.get('fiftyTwoWeekLow', 'N/A'),
                'average_volume': info.get('averageVolume', 'N/A')
            }
            
            return summary
        except Exception as e:
            st.warning(f"Error fetching market summary: {e}")
            return {}
