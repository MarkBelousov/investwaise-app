import pandas as pd
import numpy as np
try:
    import pandas_ta as ta
except ImportError:
    # Fallback if pandas_ta has compatibility issues
    ta = None

class TechnicalAnalysis:
    """Class for technical analysis calculations and indicators"""
    
    def __init__(self):
        pass
    
    def add_technical_indicators(self, data):
        """
        Add technical indicators to stock data
        
        Args:
            data (pd.DataFrame): Stock data with OHLCV columns
        
        Returns:
            pd.DataFrame: Data with technical indicators added
        """
        if data.empty or len(data) < 50:
            return data
        
        df = data.copy()
        
        try:
            # Moving Averages
            df['MA20'] = df['Close'].rolling(window=20).mean()
            df['MA50'] = df['Close'].rolling(window=50).mean()
            df['MA200'] = df['Close'].rolling(window=200).mean()
            
            # Exponential Moving Averages
            df['EMA12'] = df['Close'].ewm(span=12).mean()
            df['EMA26'] = df['Close'].ewm(span=26).mean()
            
            # MACD
            df['MACD'] = df['EMA12'] - df['EMA26']
            df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
            df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
            
            # RSI
            if ta is not None:
                df['RSI'] = ta.rsi(df['Close'], length=14)
            else:
                # Manual RSI calculation
                delta = df['Close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss
                df['RSI'] = 100 - (100 / (1 + rs))
            
            # Bollinger Bands
            if ta is not None:
                bb = ta.bbands(df['Close'], length=20, std=2)
                if bb is not None:
                    df['BB_Upper'] = bb['BBU_20_2.0']
                    df['BB_Lower'] = bb['BBL_20_2.0']
                    df['BB_Middle'] = bb['BBM_20_2.0']
            else:
                # Manual Bollinger Bands calculation
                rolling_mean = df['Close'].rolling(window=20).mean()
                rolling_std = df['Close'].rolling(window=20).std()
                df['BB_Upper'] = rolling_mean + (rolling_std * 2)
                df['BB_Lower'] = rolling_mean - (rolling_std * 2)
                df['BB_Middle'] = rolling_mean
            
            # Volume indicators
            df['Volume_MA'] = df['Volume'].rolling(window=20).mean()
            
            # Stochastic Oscillator
            if ta is not None:
                stoch = ta.stoch(df['High'], df['Low'], df['Close'])
                if stoch is not None:
                    df['Stoch_K'] = stoch['STOCHk_14_3_3']
                    df['Stoch_D'] = stoch['STOCHd_14_3_3']
            else:
                # Manual Stochastic calculation
                low_min = df['Low'].rolling(window=14).min()
                high_max = df['High'].rolling(window=14).max()
                df['Stoch_K'] = 100 * ((df['Close'] - low_min) / (high_max - low_min))
                df['Stoch_D'] = df['Stoch_K'].rolling(window=3).mean()
            
            # Average True Range (ATR)
            if ta is not None:
                df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
            else:
                # Manual ATR calculation
                df['HL'] = df['High'] - df['Low']
                df['HC'] = abs(df['High'] - df['Close'].shift(1))
                df['LC'] = abs(df['Low'] - df['Close'].shift(1))
                df['TR'] = df[['HL', 'HC', 'LC']].max(axis=1)
                df['ATR'] = df['TR'].rolling(window=14).mean()
                # Clean up temporary columns
                df.drop(['HL', 'HC', 'LC', 'TR'], axis=1, inplace=True)
            
            # Support and Resistance levels
            df = self.calculate_support_resistance(df)
            
        except Exception as e:
            print(f"Error calculating technical indicators: {e}")
        
        return df
    
    def calculate_support_resistance(self, data, window=20):
        """
        Calculate support and resistance levels
        
        Args:
            data (pd.DataFrame): Stock data
            window (int): Window for calculation
        
        Returns:
            pd.DataFrame: Data with support/resistance levels
        """
        df = data.copy()
        
        try:
            # Rolling min/max for support/resistance
            df['Support'] = df['Low'].rolling(window=window).min()
            df['Resistance'] = df['High'].rolling(window=window).max()
            
            # Pivot points
            df['Pivot'] = (df['High'] + df['Low'] + df['Close']) / 3
            df['R1'] = 2 * df['Pivot'] - df['Low']
            df['S1'] = 2 * df['Pivot'] - df['High']
            
        except Exception as e:
            print(f"Error calculating support/resistance: {e}")
        
        return df
    
    def generate_signals(self, data):
        """
        Generate buy/sell signals based on technical indicators
        
        Args:
            data (pd.DataFrame): Stock data with technical indicators
        
        Returns:
            dict: Trading signals and analysis
        """
        if data.empty or len(data) < 50:
            return {}
        
        signals = {}
        latest = data.iloc[-1]
        
        try:
            # RSI signals
            if 'RSI' in data.columns and not pd.isna(latest['RSI']):
                if latest['RSI'] > 70:
                    signals['RSI'] = {'signal': 'SELL', 'strength': 'Strong', 'reason': 'Overbought'}
                elif latest['RSI'] < 30:
                    signals['RSI'] = {'signal': 'BUY', 'strength': 'Strong', 'reason': 'Oversold'}
                else:
                    signals['RSI'] = {'signal': 'HOLD', 'strength': 'Neutral', 'reason': 'Normal range'}
            
            # MACD signals
            if all(col in data.columns for col in ['MACD', 'MACD_Signal']):
                if latest['MACD'] > latest['MACD_Signal'] and len(data) > 1:
                    prev = data.iloc[-2]
                    if prev['MACD'] <= prev['MACD_Signal']:
                        signals['MACD'] = {'signal': 'BUY', 'strength': 'Medium', 'reason': 'Bullish crossover'}
                    else:
                        signals['MACD'] = {'signal': 'HOLD', 'strength': 'Weak', 'reason': 'Above signal line'}
                else:
                    signals['MACD'] = {'signal': 'SELL', 'strength': 'Medium', 'reason': 'Below signal line'}
            
            # Moving Average signals
            if all(col in data.columns for col in ['MA20', 'MA50']):
                price = latest['Close']
                ma20 = latest['MA20']
                ma50 = latest['MA50']
                
                if price > ma20 > ma50:
                    signals['MA'] = {'signal': 'BUY', 'strength': 'Strong', 'reason': 'Price above both MAs'}
                elif price < ma20 < ma50:
                    signals['MA'] = {'signal': 'SELL', 'strength': 'Strong', 'reason': 'Price below both MAs'}
                else:
                    signals['MA'] = {'signal': 'HOLD', 'strength': 'Medium', 'reason': 'Mixed signals'}
            
            # Bollinger Bands signals
            if all(col in data.columns for col in ['BB_Upper', 'BB_Lower']):
                price = latest['Close']
                if price > latest['BB_Upper']:
                    signals['BB'] = {'signal': 'SELL', 'strength': 'Medium', 'reason': 'Above upper band'}
                elif price < latest['BB_Lower']:
                    signals['BB'] = {'signal': 'BUY', 'strength': 'Medium', 'reason': 'Below lower band'}
                else:
                    signals['BB'] = {'signal': 'HOLD', 'strength': 'Neutral', 'reason': 'Within bands'}
            
            # Volume analysis
            if 'Volume_MA' in data.columns:
                vol_ratio = latest['Volume'] / latest['Volume_MA']
                if vol_ratio > 1.5:
                    signals['Volume'] = {'signal': 'WATCH', 'strength': 'High', 'reason': 'High volume'}
                else:
                    signals['Volume'] = {'signal': 'NORMAL', 'strength': 'Low', 'reason': 'Normal volume'}
        
        except Exception as e:
            print(f"Error generating signals: {e}")
        
        return signals
    
    def calculate_volatility(self, data, window=20):
        """
        Calculate various volatility measures
        
        Args:
            data (pd.DataFrame): Stock data
            window (int): Window for calculation
        
        Returns:
            dict: Volatility metrics
        """
        if data.empty or len(data) < window:
            return {}
        
        try:
            # Calculate returns
            returns = data['Close'].pct_change().dropna()
            
            # Historical volatility (annualized)
            hist_vol = returns.rolling(window=window).std() * np.sqrt(252) * 100
            
            # Average True Range based volatility
            if 'ATR' in data.columns:
                atr_vol = (data['ATR'] / data['Close']) * 100
                current_atr_vol = atr_vol.iloc[-1] if not pd.isna(atr_vol.iloc[-1]) else 0
            else:
                current_atr_vol = 0
            
            return {
                'historical_volatility': hist_vol.iloc[-1] if not pd.isna(hist_vol.iloc[-1]) else 0,
                'atr_volatility': current_atr_vol,
                'volatility_trend': self.get_volatility_trend(hist_vol)
            }
        
        except Exception as e:
            print(f"Error calculating volatility: {e}")
            return {}
    
    def get_volatility_trend(self, volatility_series):
        """
        Determine volatility trend
        
        Args:
            volatility_series (pd.Series): Volatility time series
        
        Returns:
            str: Volatility trend
        """
        if len(volatility_series) < 10:
            return "Insufficient data"
        
        recent_avg = volatility_series.tail(5).mean()
        older_avg = volatility_series.tail(20).head(10).mean()
        
        if recent_avg > older_avg * 1.2:
            return "Increasing"
        elif recent_avg < older_avg * 0.8:
            return "Decreasing"
        else:
            return "Stable"
    
    def momentum_analysis(self, data):
        """
        Analyze price momentum
        
        Args:
            data (pd.DataFrame): Stock data
        
        Returns:
            dict: Momentum analysis
        """
        if data.empty or len(data) < 20:
            return {}
        
        try:
            # Price rate of change
            roc_10 = ((data['Close'].iloc[-1] / data['Close'].iloc[-11]) - 1) * 100
            roc_20 = ((data['Close'].iloc[-1] / data['Close'].iloc[-21]) - 1) * 100
            
            # Momentum oscillator
            momentum = data['Close'].iloc[-1] - data['Close'].iloc[-11]
            
            return {
                'roc_10_day': roc_10,
                'roc_20_day': roc_20,
                'momentum': momentum,
                'momentum_strength': self.classify_momentum(roc_20)
            }
        
        except Exception as e:
            print(f"Error in momentum analysis: {e}")
            return {}
    
    def classify_momentum(self, roc):
        """
        Classify momentum strength
        
        Args:
            roc (float): Rate of change
        
        Returns:
            str: Momentum classification
        """
        if roc > 10:
            return "Very Strong Bullish"
        elif roc > 5:
            return "Strong Bullish"
        elif roc > 2:
            return "Moderate Bullish"
        elif roc > -2:
            return "Neutral"
        elif roc > -5:
            return "Moderate Bearish"
        elif roc > -10:
            return "Strong Bearish"
        else:
            return "Very Strong Bearish"
