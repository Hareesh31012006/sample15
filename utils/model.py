import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

class StockPredictor:
    def __init__(self):
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.model = None
        self.is_trained = False
        self.feature_importance = {}
    
    def prepare_data(self, df, feature_columns, lookback_days=30):
        """Enhanced data preparation with feature engineering"""
        if df.empty or len(df) < lookback_days:
            return None, None, None
        
        # Create enhanced features
        enhanced_df = self._create_enhanced_features(df, feature_columns)
        
        # Select features
        data = enhanced_df.values
        
        # Scale the data
        scaled_data = self.scaler.fit_transform(data)
        
        X, y = [], []
        for i in range(lookback_days, len(scaled_data)):
            X.append(scaled_data[i-lookback_days:i])
            y.append(scaled_data[i, 0])  # Predict closing price
        
        return np.array(X), np.array(y), scaled_data
    
    def _create_enhanced_features(self, df, base_features):
        """Create enhanced features for better prediction accuracy"""
        enhanced_data = df[base_features].copy()
        
        # Add price momentum features
        if 'Close' in df.columns:
            # Price changes
            enhanced_data['price_change_1d'] = df['Close'].pct_change(1)
            enhanced_data['price_change_3d'] = df['Close'].pct_change(3)
            enhanced_data['price_change_7d'] = df['Close'].pct_change(7)
            
            # Volatility
            enhanced_data['volatility_7d'] = df['Close'].pct_change().rolling(7).std()
            
            # High-Low range
            if 'High' in df.columns and 'Low' in df.columns:
                enhanced_data['hl_range'] = (df['High'] - df['Low']) / df['Close']
        
        # Add volume features
        if 'Volume' in df.columns:
            enhanced_data['volume_change'] = df['Volume'].pct_change()
            enhanced_data['volume_sma_ratio'] = df['Volume'] / df['Volume'].rolling(20).mean()
        
        # Fill NaN values with forward then backward fill
        enhanced_data = enhanced_data.fillna(method='ffill').fillna(method='bfill')
        
        return enhanced_data
    
    def train_enhanced_model(self, df, feature_columns=['Close', 'Volume', 'rsi', 'macd']):
        """Enhanced model training with feature selection"""
        try:
            # Create enhanced features
            features = self._create_enhanced_features(df, feature_columns)
            target = features['Close'].shift(-1).dropna()
            features = features.iloc[:-1]
            
            if len(features) < 15:  # Increased minimum data requirement
                return False
            
            # Use Random Forest for better accuracy (falls back to Linear Regression)
            try:
                self.model = RandomForestRegressor(
                    n_estimators=50,  # Reduced for speed but better than linear
                    max_depth=10,
                    random_state=42,
                    n_jobs=-1
                )
                self.model.fit(features, target)
                
                # Store feature importance
                for idx, col in enumerate(features.columns):
                    self.feature_importance[col] = self.model.feature_importances_[idx]
                    
            except Exception as e:
                print(f"Random Forest failed, using Linear Regression: {e}")
                self.model = LinearRegression()
                self.model.fit(features, target)
            
            self.is_trained = True
            return True
            
        except Exception as e:
            print(f"Error training enhanced model: {e}")
            return False
    
    def predict_next_day(self, df, feature_columns=['Close', 'Volume', 'rsi', 'macd']):
        """Enhanced prediction with confidence estimation"""
        if not self.is_trained or self.model is None:
            return None, None, None
        
        try:
            # Create enhanced features for prediction
            features = self._create_enhanced_features(df, feature_columns)
            latest_features = features.iloc[-1:].values
            
            # Make prediction
            prediction = self.model.predict(latest_features)[0]
            
            # Calculate current price
            current_price = df['Close'].iloc[-1]
            
            # Simple confidence estimation based on recent volatility
            recent_volatility = df['Close'].pct_change().tail(10).std()
            confidence = max(0.1, 1.0 - (recent_volatility * 10))  # Higher volatility = lower confidence
            
            return prediction, current_price, confidence
            
        except Exception as e:
            print(f"Error making prediction: {e}")
            return None, None, None
    
    def calculate_enhanced_metrics(self, df, feature_columns):
        """Calculate enhanced prediction metrics"""
        try:
            # Create enhanced features
            features = self._create_enhanced_features(df, feature_columns)
            features = features.iloc[:-1]  # Remove last row which has no target
            target = df['Close'].shift(-1).dropna().iloc[:-1]  # Align with features
            
            if len(features) < 10:
                return None
            
            # Train-test split (80-20)
            split_idx = int(len(features) * 0.8)
            X_train, X_test = features.iloc[:split_idx], features.iloc[split_idx:]
            y_train, y_test = target.iloc[:split_idx], target.iloc[split_idx:]
            
            # Train temporary model
            temp_model = LinearRegression()
            temp_model.fit(X_train, y_train)
            
            # Predictions
            y_pred = temp_model.predict(X_test)
            
            # Calculate metrics
            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            
            # R-squared score
            from sklearn.metrics import r2_score
            r2 = r2_score(y_test, y_pred)
            
            return {
                'MAE': mae,
                'MSE': mse,
                'RMSE': rmse,
                'R2_Score': r2,
                'Accuracy_Score': max(0, r2)  # Convert to 0-1 scale
            }
            
        except Exception as e:
            print(f"Error calculating enhanced metrics: {e}")
            return None

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
