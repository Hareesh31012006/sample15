import yfinance as yf
import pandas as pd
import numpy as np
from alpha_vantage.timeseries import TimeSeries
import requests
import os
from datetime import datetime, timedelta

class DataFetcher:
    def __init__(self):
        self.alpha_vantage_key = os.getenv('ALPHA_VANTAGE_API_KEY', 'demo')
        self.newsapi_key = os.getenv('NEWS_API_KEY', 'your_news_api_key')
        
    def get_stock_data_yahoo(self, symbol, period='6mo'):
        """Fetch stock data from Yahoo Finance"""
        try:
            stock = yf.Ticker(symbol)
            data = stock.history(period=period)
            return data
        except Exception as e:
            print(f"Error fetching data from Yahoo: {e}")
            return None
    
    def get_stock_data_alpha_vantage(self, symbol):
        """Fetch stock data from Alpha Vantage"""
        try:
            ts = TimeSeries(key=self.alpha_vantage_key, output_format='pandas')
            data, meta_data = ts.get_daily(symbol=symbol, outputsize='compact')
            return data
        except Exception as e:
            print(f"Error fetching data from Alpha Vantage: {e}")
            return None
    
    def get_news_data(self, symbol, days=7):
        """Fetch news data for a stock symbol"""
        try:
            from_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
            url = f"https://newsapi.org/v2/everything?q={symbol}&from={from_date}&sortBy=publishedAt&apiKey={self.newsapi_key}"
            
            response = requests.get(url)
            articles = response.json().get('articles', [])
            
            news_data = []
            for article in articles[:20]:  # Limit to 20 articles
                news_data.append({
                    'title': article['title'],
                    'description': article['description'],
                    'published_at': article['publishedAt'],
                    'source': article['source']['name']
                })
            
            return pd.DataFrame(news_data)
        except Exception as e:
            print(f"Error fetching news: {e}")
            return pd.DataFrame()
    
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
