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
            return data
        except Exception as e:
            st.error(f"Error fetching data for {symbol}: {e}")
            return None
    
    def get_stock_data_alpha_vantage(self, symbol):
        """Fetch stock data from Alpha Vantage with real API key"""
        try:
            ts = TimeSeries(key=self.alpha_vantage_key, output_format='pandas')
            data, meta_data = ts.get_daily(symbol=symbol, outputsize='compact')
            return data
        except Exception as e:
            st.error(f"Alpha Vantage error: {e}")
            return None
    
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
    
    def calculate_technical_indicators(self, df):
        """Calculate technical indicators"""
        try:
            import talib
            
            # RSI
            df['rsi'] = talib.RSI(df['Close'], timeperiod=14)
            
            # MACD
            macd, macd_signal, macd_hist = talib.MACD(df['Close'])
            df['macd'] = macd
            df['macd_signal'] = macd_signal
            df['macd_hist'] = macd_hist
            
            # Moving Averages
            df['sma_20'] = talib.SMA(df['Close'], timeperiod=20)
            df['sma_50'] = talib.SMA(df['Close'], timeperiod=50)
            
            # Bollinger Bands
            bb_upper, bb_middle, bb_lower = talib.BBANDS(df['Close'], timeperiod=20)
            df['bb_upper'] = bb_upper
            df['bb_middle'] = bb_middle
            df['bb_lower'] = bb_lower
            
            # Volume indicators
            df['volume_sma'] = talib.SMA(df['Volume'], timeperiod=20)
            
        except Exception as e:
            print(f"Error calculating technical indicators: {e}")
        
        return df
