"""
Feature Engineering Pipeline for Forex Trading Bot

This module creates comprehensive feature sets by applying technical indicators
across multiple timeframes and generating advanced trading features.

Features include:
- Multi-timeframe technical indicators
- Price pattern recognition
- Market microstructure features
- Statistical features
- Time-based features
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import logging
from datetime import datetime, timezone
import warnings

from technical_indicators import TechnicalIndicators, IndicatorResult
from config import TradingConfig
from database import Base, PriceData, TechnicalIndicator
from logger import TradingLogger

logger = logging.getLogger(__name__)


@dataclass
class FeatureSet:
    """Container for engineered features."""
    features: pd.DataFrame
    feature_names: List[str]
    metadata: Dict
    timestamp: datetime


class FeatureEngineer:
    """
    Advanced feature engineering pipeline for forex trading.
    
    Creates comprehensive feature sets from price data using:
    - Technical indicators across multiple timeframes
    - Price patterns and relationships
    - Statistical features
    - Market structure features
    """
    
    def __init__(self, config: Optional[TradingConfig] = None):
        """
        Initialize FeatureEngineer.
        
        Args:
            config: Trading bot configuration
        """
        self.config = config or TradingConfig()
        self.indicators = TechnicalIndicators()
        self.logger = TradingLogger("feature_engineer")
        
        # Define timeframes for multi-timeframe analysis
        self.timeframes = {
            "1m": 1,
            "5m": 5,
            "15m": 15,
            "1h": 60,
            "4h": 240,
            "1d": 1440
        }
        
        # Technical indicator configurations
        self.indicator_configs = {
            "sma": [5, 10, 20, 50, 100, 200],
            "ema": [5, 10, 12, 20, 26, 50],
            "rsi": [14, 21],
            "macd": [(12, 26, 9)],
            "bb": [(20, 2.0), (20, 2.5)],
            "atr": [14, 21],
            "stochastic": [(14, 3), (21, 3)],
            "williams_r": [14, 21],
            "cci": [20]
        }
    
    def prepare_data(self, data: pd.DataFrame, validate: bool = True) -> pd.DataFrame:
        """
        Prepare and validate price data for feature engineering.
        
        Args:
            data: DataFrame with OHLCV columns
            validate: Whether to perform data validation
            
        Returns:
            Validated and prepared DataFrame
        """
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        
        if validate:
            # Check required columns
            missing_columns = [col for col in required_columns if col not in data.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            # Check for sufficient data
            if len(data) < 200:
                self.logger.log_signal("WARNING", 
                    f"Limited data available: {len(data)} bars. May affect indicator quality.")
            
            # Check for data quality issues
            null_counts = data[required_columns].isnull().sum()
            if null_counts.any():
                self.logger.log_signal("WARNING", 
                    f"Found null values in data: {null_counts.to_dict()}")
        
        # Ensure proper sorting by timestamp if available
        if 'timestamp' in data.columns:
            data = data.sort_values('timestamp')
        
        # Reset index to ensure clean indexing
        data = data.reset_index(drop=True)
        
        return data.copy()
    
    def generate_basic_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate basic price-derived features.
        
        Args:
            data: OHLCV DataFrame
            
        Returns:
            DataFrame with basic features added
        """
        features = data.copy()
        
        # Price relationships
        features['hl_ratio'] = (data['high'] - data['low']) / data['close']
        features['oc_ratio'] = (data['close'] - data['open']) / data['open']
        features['body_size'] = np.abs(data['close'] - data['open']) / data['close']
        features['upper_shadow'] = (data['high'] - np.maximum(data['open'], data['close'])) / data['close']
        features['lower_shadow'] = (np.minimum(data['open'], data['close']) - data['low']) / data['close']
        
        # Price changes and returns
        features['price_change'] = data['close'].pct_change()
        features['log_returns'] = np.log(data['close'] / data['close'].shift(1))
        features['high_low_pct'] = (data['high'] - data['low']) / data['low']
        
        # Volume features (if available)
        if 'volume' in data.columns and not data['volume'].isna().all():
            features['volume_sma'] = data['volume'].rolling(20).mean()
            features['volume_ratio'] = data['volume'] / features['volume_sma']
            features['price_volume'] = data['close'] * data['volume']
        
        # Volatility measures
        features['volatility_5'] = features['log_returns'].rolling(5).std()
        features['volatility_20'] = features['log_returns'].rolling(20).std()
        
        return features
    
    def generate_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate all configured technical indicators.
        
        Args:
            data: OHLCV DataFrame
            
        Returns:
            DataFrame with technical indicators added
        """
        features = data.copy()
        
        self.logger.log_signal("INFO", "Generating technical indicators...")
        
        # Simple Moving Averages
        for period in self.indicator_configs["sma"]:
            try:
                sma_result = self.indicators.sma(data['close'], period)
                features[f'sma_{period}'] = sma_result.values
                
                # SMA-based features
                features[f'price_sma_{period}_ratio'] = data['close'] / features[f'sma_{period}']
                features[f'sma_{period}_slope'] = features[f'sma_{period}'].diff(5)
            except Exception as e:
                self.logger.log_signal("ERROR", f"Failed to calculate SMA({period}): {e}")
        
        # Exponential Moving Averages
        for period in self.indicator_configs["ema"]:
            try:
                ema_result = self.indicators.ema(data['close'], period)
                features[f'ema_{period}'] = ema_result.values
                
                # EMA-based features
                features[f'price_ema_{period}_ratio'] = data['close'] / features[f'ema_{period}']
                features[f'ema_{period}_slope'] = features[f'ema_{period}'].diff(5)
            except Exception as e:
                self.logger.log_signal("ERROR", f"Failed to calculate EMA({period}): {e}")
        
        # RSI
        for period in self.indicator_configs["rsi"]:
            try:
                rsi_result = self.indicators.rsi(data['close'], period)
                features[f'rsi_{period}'] = rsi_result.values
                
                # RSI-based features
                features[f'rsi_{period}_oversold'] = (features[f'rsi_{period}'] < 30).astype(int)
                features[f'rsi_{period}_overbought'] = (features[f'rsi_{period}'] > 70).astype(int)
            except Exception as e:
                self.logger.log_signal("ERROR", f"Failed to calculate RSI({period}): {e}")
        
        # MACD
        for fast, slow, signal in self.indicator_configs["macd"]:
            try:
                macd_result = self.indicators.macd(data['close'], fast, slow, signal)
                features[f'macd_{fast}_{slow}'] = macd_result.values[:, 0]
                features[f'macd_signal_{fast}_{slow}_{signal}'] = macd_result.values[:, 1]
                features[f'macd_histogram_{fast}_{slow}_{signal}'] = macd_result.values[:, 2]
                
                # MACD-based features
                features[f'macd_bullish_{fast}_{slow}'] = (
                    features[f'macd_{fast}_{slow}'] > features[f'macd_signal_{fast}_{slow}_{signal}']
                ).astype(int)
            except Exception as e:
                self.logger.log_signal("ERROR", f"Failed to calculate MACD({fast},{slow},{signal}): {e}")
        
        # Bollinger Bands
        for period, std_dev in self.indicator_configs["bb"]:
            try:
                bb_result = self.indicators.bollinger_bands(data['close'], period, std_dev)
                features[f'bb_upper_{period}_{std_dev}'] = bb_result.values[:, 0]
                features[f'bb_middle_{period}_{std_dev}'] = bb_result.values[:, 1]
                features[f'bb_lower_{period}_{std_dev}'] = bb_result.values[:, 2]
                
                # Bollinger Band features
                features[f'bb_position_{period}_{std_dev}'] = (
                    (data['close'] - features[f'bb_lower_{period}_{std_dev}']) /
                    (features[f'bb_upper_{period}_{std_dev}'] - features[f'bb_lower_{period}_{std_dev}'])
                )
                features[f'bb_squeeze_{period}_{std_dev}'] = (
                    features[f'bb_upper_{period}_{std_dev}'] - features[f'bb_lower_{period}_{std_dev}']
                ) / features[f'bb_middle_{period}_{std_dev}']
            except Exception as e:
                self.logger.log_signal("ERROR", f"Failed to calculate Bollinger Bands({period},{std_dev}): {e}")
        
        # ATR
        for period in self.indicator_configs["atr"]:
            try:
                atr_result = self.indicators.atr(data['high'], data['low'], data['close'], period)
                features[f'atr_{period}'] = atr_result.values
                
                # ATR-based features
                features[f'atr_{period}_normalized'] = features[f'atr_{period}'] / data['close']
            except Exception as e:
                self.logger.log_signal("ERROR", f"Failed to calculate ATR({period}): {e}")
        
        # Stochastic
        for k_period, d_period in self.indicator_configs["stochastic"]:
            try:
                stoch_result = self.indicators.stochastic(
                    data['high'], data['low'], data['close'], k_period, d_period
                )
                features[f'stoch_k_{k_period}_{d_period}'] = stoch_result.values[:, 0]
                features[f'stoch_d_{k_period}_{d_period}'] = stoch_result.values[:, 1]
                
                # Stochastic features
                features[f'stoch_oversold_{k_period}'] = (features[f'stoch_k_{k_period}_{d_period}'] < 20).astype(int)
                features[f'stoch_overbought_{k_period}'] = (features[f'stoch_k_{k_period}_{d_period}'] > 80).astype(int)
            except Exception as e:
                self.logger.log_signal("ERROR", f"Failed to calculate Stochastic({k_period},{d_period}): {e}")
        
        # Williams %R
        for period in self.indicator_configs["williams_r"]:
            try:
                wr_result = self.indicators.williams_r(data['high'], data['low'], data['close'], period)
                features[f'williams_r_{period}'] = wr_result.values
            except Exception as e:
                self.logger.log_signal("ERROR", f"Failed to calculate Williams %R({period}): {e}")
        
        # CCI
        for period in self.indicator_configs["cci"]:
            try:
                cci_result = self.indicators.cci(data['high'], data['low'], data['close'], period)
                features[f'cci_{period}'] = cci_result.values
            except Exception as e:
                self.logger.log_signal("ERROR", f"Failed to calculate CCI({period}): {e}")
        
        return features
    
    def generate_pattern_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate price pattern recognition features.
        
        Args:
            data: OHLCV DataFrame
            
        Returns:
            DataFrame with pattern features added
        """
        features = data.copy()
        
        # Candlestick patterns (simplified)
        body = np.abs(data['close'] - data['open'])
        upper_shadow = data['high'] - np.maximum(data['close'], data['open'])
        lower_shadow = np.minimum(data['close'], data['open']) - data['low']
        
        # Doji pattern (small body relative to shadows)
        features['doji'] = (body < (upper_shadow + lower_shadow) * 0.1).astype(int)
        
        # Hammer pattern (small body, long lower shadow, small upper shadow)
        features['hammer'] = (
            (lower_shadow > body * 2) & 
            (upper_shadow < body * 0.5) & 
            (body > 0)
        ).astype(int)
        
        # Shooting star (small body, long upper shadow, small lower shadow)
        features['shooting_star'] = (
            (upper_shadow > body * 2) & 
            (lower_shadow < body * 0.5) & 
            (body > 0)
        ).astype(int)
        
        # Engulfing patterns (simplified)
        bullish_engulfing = (
            (data['close'] > data['open']) & 
            (data['close'].shift(1) < data['open'].shift(1)) &
            (data['open'] < data['close'].shift(1)) & 
            (data['close'] > data['open'].shift(1))
        )
        features['bullish_engulfing'] = bullish_engulfing.astype(int)
        
        bearish_engulfing = (
            (data['close'] < data['open']) & 
            (data['close'].shift(1) > data['open'].shift(1)) &
            (data['open'] > data['close'].shift(1)) & 
            (data['close'] < data['open'].shift(1))
        )
        features['bearish_engulfing'] = bearish_engulfing.astype(int)
        
        # Gap detection
        features['gap_up'] = (data['low'] > data['high'].shift(1)).astype(int)
        features['gap_down'] = (data['high'] < data['low'].shift(1)).astype(int)
        
        # Support and resistance levels (simplified)
        window = 20
        features['local_high'] = (
            data['high'] == data['high'].rolling(window, center=True).max()
        ).astype(int)
        features['local_low'] = (
            data['low'] == data['low'].rolling(window, center=True).min()
        ).astype(int)
        
        return features
    
    def generate_statistical_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate statistical and momentum features.
        
        Args:
            data: DataFrame with price data
            
        Returns:
            DataFrame with statistical features added
        """
        features = data.copy()
        
        # Rolling statistics
        for window in [5, 10, 20]:
            features[f'mean_reversion_{window}'] = (
                (data['close'] - data['close'].rolling(window).mean()) / 
                data['close'].rolling(window).std()
            )
            
            features[f'momentum_{window}'] = data['close'] / data['close'].shift(window) - 1
            
            features[f'volatility_{window}'] = (
                data['close'].pct_change().rolling(window).std() * np.sqrt(window)
            )
            
            # Price position within recent range
            rolling_high = data['high'].rolling(window).max()
            rolling_low = data['low'].rolling(window).min()
            features[f'price_position_{window}'] = (
                (data['close'] - rolling_low) / (rolling_high - rolling_low)
            )
        
        # Autocorrelation features
        returns = data['close'].pct_change()
        for lag in [1, 2, 5]:
            features[f'return_autocorr_lag{lag}'] = (
                returns.rolling(20).apply(lambda x: np.corrcoef(x[:-lag], x[lag:])[0,1])
            )
        
        return features
    
    def generate_time_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate time-based features if timestamp is available.
        
        Args:
            data: DataFrame with timestamp column
            
        Returns:
            DataFrame with time features added
        """
        features = data.copy()
        
        if 'timestamp' in data.columns:
            timestamp = pd.to_datetime(data['timestamp'])
            
            # Time components
            features['hour'] = timestamp.dt.hour
            features['day_of_week'] = timestamp.dt.dayofweek
            features['month'] = timestamp.dt.month
            
            # Trading session features (assuming UTC timestamps)
            # Asian session: 23:00-08:00 UTC
            features['asian_session'] = (
                (features['hour'] >= 23) | (features['hour'] <= 8)
            ).astype(int)
            
            # European session: 07:00-16:00 UTC
            features['european_session'] = (
                (features['hour'] >= 7) & (features['hour'] <= 16)
            ).astype(int)
            
            # US session: 13:00-22:00 UTC
            features['us_session'] = (
                (features['hour'] >= 13) & (features['hour'] <= 22)
            ).astype(int)
            
            # Session overlaps
            features['eur_us_overlap'] = (
                (features['hour'] >= 13) & (features['hour'] <= 16)
            ).astype(int)
            
            features['asia_eur_overlap'] = (
                (features['hour'] >= 7) & (features['hour'] <= 8)
            ).astype(int)
        
        return features
    
    def create_feature_set(self, data: pd.DataFrame, include_patterns: bool = True, 
                          include_time: bool = True) -> FeatureSet:
        """
        Create comprehensive feature set from price data.
        
        Args:
            data: OHLCV DataFrame
            include_patterns: Whether to include pattern features
            include_time: Whether to include time features
            
        Returns:
            FeatureSet object with engineered features
        """
        self.logger.log_signal("INFO", "Starting feature engineering pipeline...")
        
        # Prepare data
        prepared_data = self.prepare_data(data)
        
        # Generate feature sets
        features = prepared_data
        features = self.generate_basic_features(features)
        features = self.generate_technical_indicators(features)
        features = self.generate_statistical_features(features)
        
        if include_patterns:
            features = self.generate_pattern_features(features)
        
        if include_time and 'timestamp' in data.columns:
            features = self.generate_time_features(features)
        
        # Get feature names (exclude original OHLCV columns)
        original_columns = set(['open', 'high', 'low', 'close', 'volume', 'timestamp'])
        feature_names = [col for col in features.columns if col not in original_columns]
        
        # Calculate feature statistics
        feature_data = features[feature_names]
        null_counts = feature_data.isnull().sum()
        completeness = (1 - null_counts / len(feature_data)) * 100
        
        metadata = {
            "total_features": len(feature_names),
            "total_rows": len(features),
            "completeness_percent": completeness.mean(),
            "null_features": (null_counts > 0).sum(),
            "feature_categories": {
                "basic": len([f for f in feature_names if any(x in f for x in ['ratio', 'change', 'returns', 'shadow'])]),
                "technical": len([f for f in feature_names if any(x in f for x in ['sma', 'ema', 'rsi', 'macd', 'bb', 'atr'])]),
                "patterns": len([f for f in feature_names if any(x in f for x in ['doji', 'hammer', 'engulfing', 'gap'])]),
                "statistical": len([f for f in feature_names if any(x in f for x in ['momentum', 'volatility', 'autocorr'])]),
                "time": len([f for f in feature_names if any(x in f for x in ['hour', 'session', 'overlap'])])
            }
        }
        
        self.logger.log_signal("SUCCESS", 
            f"Feature engineering complete: {metadata['total_features']} features generated "
            f"with {metadata['completeness_percent']:.1f}% completeness")
        
        return FeatureSet(
            features=features,
            feature_names=feature_names,
            metadata=metadata,
            timestamp=datetime.now(timezone.utc)
        )
    
    def calculate_feature_importance(self, features: pd.DataFrame, target: pd.Series, 
                                   method: str = "correlation") -> pd.Series:
        """
        Calculate feature importance scores.
        
        Args:
            features: Feature DataFrame
            target: Target variable (e.g., future returns)
            method: Method to calculate importance ("correlation", "mutual_info")
            
        Returns:
            Series with feature importance scores
        """
        if method == "correlation":
            # Calculate correlation with target
            correlations = features.corrwith(target).abs()
            return correlations.sort_values(ascending=False)
        
        elif method == "mutual_info":
            try:
                from sklearn.feature_selection import mutual_info_regression
                # Remove NaN values
                clean_data = pd.concat([features, target], axis=1).dropna()
                if len(clean_data) > 100:  # Need sufficient data
                    X = clean_data[features.columns]
                    y = clean_data[target.name]
                    scores = mutual_info_regression(X, y, random_state=42)
                    return pd.Series(scores, index=features.columns).sort_values(ascending=False)
            except ImportError:
                self.logger.log_signal("WARNING", "sklearn not available, falling back to correlation")
                return self.calculate_feature_importance(features, target, "correlation")
        
        return pd.Series(index=features.columns)
    
    def generate_targets(self, data: pd.DataFrame, horizon: int = 1, 
                        target_type: str = "returns") -> pd.Series:
        """
        Generate target variables for supervised learning.
        
        Args:
            data: Price DataFrame
            horizon: Prediction horizon in periods
            target_type: Type of target ("returns", "direction", "volatility")
            
        Returns:
            Target series
        """
        if target_type == "returns":
            # Future log returns
            future_returns = np.log(data['close'].shift(-horizon) / data['close'])
            return future_returns
        
        elif target_type == "direction":
            # Direction of future price movement (1 = up, 0 = down)
            future_prices = data['close'].shift(-horizon)
            direction = (future_prices > data['close']).astype(int)
            return direction
        
        elif target_type == "volatility":
            # Future volatility (rolling standard deviation of returns)
            returns = data['close'].pct_change()
            future_vol = returns.shift(-horizon).rolling(horizon).std()
            return future_vol
        
        else:
            raise ValueError(f"Unknown target type: {target_type}")


def create_sample_feature_pipeline():
    """Create and demonstrate feature engineering pipeline."""
    from data_simulator import ForexDataSimulator
    
    print("Feature Engineering Pipeline Demonstration")
    print("=" * 60)
    
    # Create sample data
    simulator = ForexDataSimulator()
    sample_data = simulator.generate_realistic_data("EURUSD", days=100, freq_minutes=1)
    
    # Initialize feature engineer
    feature_engineer = FeatureEngineer()
    
    # Create feature set
    feature_set = feature_engineer.create_feature_set(sample_data)
    
    print(f"Feature Engineering Results:")
    print(f"- Total features: {feature_set.metadata['total_features']}")
    print(f"- Data completeness: {feature_set.metadata['completeness_percent']:.1f}%")
    print(f"- Feature categories: {feature_set.metadata['feature_categories']}")
    
    # Generate targets and calculate feature importance
    targets = feature_engineer.generate_targets(sample_data, horizon=5, target_type="returns")
    
    # Get feature importance
    feature_data = feature_set.features[feature_set.feature_names]
    importance_scores = feature_engineer.calculate_feature_importance(feature_data, targets)
    
    print(f"\nTop 10 Most Important Features:")
    for i, (feature, score) in enumerate(importance_scores.head(10).items(), 1):
        print(f"{i:2d}. {feature:30s} {score:.4f}")
    
    print(f"\nFeature engineering pipeline completed successfully!")
    
    return feature_set


if __name__ == "__main__":
    # Run demonstration
    create_sample_feature_pipeline()