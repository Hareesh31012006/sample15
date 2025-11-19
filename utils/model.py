import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

class StockPredictor:
    def __init__(self):
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.model = None
        self.is_trained = False
    
    def prepare_data(self, df, feature_columns, lookback_days=30):
        """Prepare data for training/prediction"""
        if df.empty or len(df) < lookback_days:
            return None, None, None
        
        # Select features
        data = df[feature_columns].values
        
        # Scale the data
        scaled_data = self.scaler.fit_transform(data)
        
        X, y = [], []
        for i in range(lookback_days, len(scaled_data)):
            X.append(scaled_data[i-lookback_days:i])
            y.append(scaled_data[i, 0])  # Predict closing price
        
        return np.array(X), np.array(y), scaled_data
    
    def train_linear_regression(self, df, feature_columns=['Close', 'Volume', 'rsi', 'macd']):
        """Train a linear regression model"""
        try:
            # Use simpler approach for linear regression
            features = df[feature_columns].dropna()
            target = features['Close'].shift(-1).dropna()
            features = features.iloc[:-1]
            
            if len(features) < 10:
                return False
            
            self.model = LinearRegression()
            self.model.fit(features, target)
            self.is_trained = True
            return True
            
        except Exception as e:
            print(f"Error training model: {e}")
            return False
    
    def predict_next_day(self, df, feature_columns=['Close', 'Volume', 'rsi', 'macd']):
        """Predict next day's closing price"""
        if not self.is_trained or self.model is None:
            return None, None
        
        try:
            # Get the latest features
            latest_features = df[feature_columns].iloc[-1:].values
            
            # Make prediction
            prediction = self.model.predict(latest_features)[0]
            
            # Calculate current price
            current_price = df['Close'].iloc[-1]
            
            return prediction, current_price
            
        except Exception as e:
            print(f"Error making prediction: {e}")
            return None, None
    
    def calculate_metrics(self, y_true, y_pred):
        """Calculate prediction metrics"""
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        
        return {
            'MAE': mae,
            'MSE': mse,
            'RMSE': rmse
        }

class LSTMPredictor(nn.Module):
    """LSTM Model for time series prediction"""
    def __init__(self, input_size=4, hidden_size=50, num_layers=2, output_size=1):
        super(LSTMPredictor, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out
