import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

class StockVisualizer:
    def __init__(self):
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def plot_stock_price(self, df, title="Stock Price Chart"):
        """Create interactive stock price chart"""
        fig = go.Figure()
        
        # Candlestick chart
        fig.add_trace(go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name='Price'
        ))
        
        # Add moving averages
        if 'sma_20' in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index,
                y=df['sma_20'],
                line=dict(color='orange', width=1),
                name='SMA 20'
            ))
        
        if 'sma_50' in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index,
                y=df['sma_50'],
                line=dict(color='red', width=1),
                name='SMA 50'
            ))
        
        fig.update_layout(
            title=title,
            xaxis_title='Date',
            yaxis_title='Price ($)',
            template='plotly_white',
            height=500
        )
        
        return fig
    
    def plot_technical_indicators(self, df):
        """Plot technical indicators"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # RSI
        if 'rsi' in df.columns:
            ax1.plot(df.index, df['rsi'], label='RSI', color='purple')
            ax1.axhline(70, linestyle='--', alpha=0.5, color='red', label='Overbought')
            ax1.axhline(30, linestyle='--', alpha=0.5, color='green', label='Oversold')
            ax1.set_title('RSI Indicator')
            ax1.legend()
        
        # MACD
        if all(col in df.columns for col in ['macd', 'macd_signal']):
            ax2.plot(df.index, df['macd'], label='MACD', color='blue')
            ax2.plot(df.index, df['macd_signal'], label='Signal', color='red')
            ax2.set_title('MACD Indicator')
            ax2.legend()
        
        # Volume
        if 'Volume' in df.columns:
            ax3.bar(df.index, df['Volume'], alpha=0.7, color='gray')
            if 'volume_sma' in df.columns:
                ax3.plot(df.index, df['volume_sma'], color='red', label='Volume SMA')
            ax3.set_title('Trading Volume')
            ax3.legend()
        
        # Price with Bollinger Bands
        if all(col in df.columns for col in ['bb_upper', 'bb_middle', 'bb_lower']):
            ax4.plot(df.index, df['Close'], label='Close Price', color='black')
            ax4.plot(df.index, df['bb_upper'], label='Upper Band', linestyle='--', alpha=0.7)
            ax4.plot(df.index, df['bb_middle'], label='Middle Band', linestyle='--', alpha=0.7)
            ax4.plot(df.index, df['bb_lower'], label='Lower Band', linestyle='--', alpha=0.7)
            ax4.fill_between(df.index, df['bb_upper'], df['bb_lower'], alpha=0.1)
            ax4.set_title('Bollinger Bands')
            ax4.legend()
        
        plt.tight_layout()
        return fig
    
    def plot_sentiment_analysis(self, sentiment_score, news_df):
        """Plot sentiment analysis results"""
        fig = go.Figure()
        
        # Sentiment gauge
        fig.add_trace(go.Indicator(
            mode="gauge+number+delta",
            value=sentiment_score,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Market Sentiment"},
            delta={'reference': 0},
            gauge={
                'axis': {'range': [-1, 1]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [-1, -0.3], 'color': "red"},
                    {'range': [-0.3, 0.3], 'color': "yellow"},
                    {'range': [0.3, 1], 'color': "green"}
                ],
                'threshold': {
                    'line': {'color': "black", 'width': 4},
                    'thickness': 0.75,
                    'value': sentiment_score
                }
            }
        ))
        
        fig.update_layout(height=300)
        return fig
