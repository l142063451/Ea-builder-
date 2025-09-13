"""
Enhanced Baseline Models for Forex Trading Bot

Improved implementation with better feature engineering, data preprocessing,
and enhanced model algorithms to achieve the 60%+ accuracy target.
"""

import sqlite3
import json
import math
import statistics
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
import os

logger = logging.getLogger(__name__)


@dataclass
class EnhancedModelResult:
    """Enhanced result container for model predictions and performance."""
    model_name: str
    predictions: List[int]
    probabilities: List[float]
    actual: List[int]
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    directional_accuracy: float
    profitable_trade_rate: float
    metadata: Dict[str, Any]


class FeatureEngineer:
    """Enhanced feature engineering for better predictive power."""
    
    @staticmethod
    def create_enhanced_features(data: List[Dict]) -> Tuple[List[List[float]], List[str]]:
        """Create enhanced feature set with better predictive power."""
        logger.info("Creating enhanced feature set...")
        
        features = []
        feature_names = []
        
        for i in range(len(data)):
            feature_vector = []
            
            # Basic price features
            feature_vector.extend([
                data[i]['open'],
                data[i]['high'], 
                data[i]['low'],
                data[i]['close'],
                math.log(data[i]['volume']) if data[i]['volume'] > 0 else 0
            ])
            
            # Price relationships
            if i > 0:
                prev_close = data[i-1]['close']
                price_change = (data[i]['close'] - prev_close) / prev_close
                feature_vector.append(price_change)
                
                # High-Low spread
                hl_spread = (data[i]['high'] - data[i]['low']) / data[i]['close']
                feature_vector.append(hl_spread)
                
                # Open-Close relationship
                oc_change = (data[i]['close'] - data[i]['open']) / data[i]['open']
                feature_vector.append(oc_change)
            else:
                feature_vector.extend([0, 0, 0])
            
            # Technical indicators with null handling
            feature_vector.extend([
                data[i].get('sma_20') or data[i]['close'],
                data[i].get('ema_20') or data[i]['close'],
                (data[i].get('rsi_14') or 50) / 100,  # Normalize RSI
                (data[i].get('macd_line') or 0) * 10000,  # Scale MACD
                data[i].get('bb_upper') or (data[i]['close'] * 1.02),
                data[i].get('bb_lower') or (data[i]['close'] * 0.98),
                (data[i].get('atr_14') or 0) * 10000,  # Scale ATR
                (data[i].get('stoch_k') or 50) / 100,  # Normalize Stochastic
                (data[i].get('stoch_d') or 50) / 100,
                ((data[i].get('williams_r') or -50) + 100) / 100,  # Normalize Williams %R
                (data[i].get('cci_20') or 0) / 100  # Normalize CCI
            ])
            
            # Derived features
            close = data[i]['close']
            sma_20 = data[i].get('sma_20') or close
            ema_20 = data[i].get('ema_20') or close
            bb_upper = data[i].get('bb_upper') or (close * 1.02)
            bb_lower = data[i].get('bb_lower') or (close * 0.98)
            
            # Price position relative to indicators
            feature_vector.extend([
                (close - sma_20) / sma_20,  # Price vs SMA
                (close - ema_20) / ema_20,  # Price vs EMA
                (close - bb_upper) / close,  # Price vs BB upper
                (close - bb_lower) / close,  # Price vs BB lower
            ])
            
            # Momentum features (if we have previous data)
            if i >= 5:
                # 5-period momentum
                momentum_5 = (close - data[i-5]['close']) / data[i-5]['close']
                feature_vector.append(momentum_5)
                
                # Price acceleration (2nd derivative)
                if i >= 10:
                    momentum_10 = (close - data[i-10]['close']) / data[i-10]['close']
                    acceleration = momentum_5 - momentum_10
                    feature_vector.append(acceleration)
                else:
                    feature_vector.append(0)
                    
                # Volatility (5-period standard deviation of returns)
                returns = []
                for j in range(max(0, i-4), i+1):
                    if j > 0:
                        ret = (data[j]['close'] - data[j-1]['close']) / data[j-1]['close']
                        returns.append(ret)
                
                if len(returns) > 1:
                    volatility = statistics.stdev(returns)
                    feature_vector.append(volatility * 100)  # Scale volatility
                else:
                    feature_vector.append(0)
            else:
                feature_vector.extend([0, 0, 0])  # Momentum, acceleration, volatility
            
            # RSI-based features
            rsi = data[i].get('rsi_14') or 50
            feature_vector.extend([
                1 if rsi > 70 else 0,  # Overbought
                1 if rsi < 30 else 0,  # Oversold
                abs(rsi - 50) / 50     # RSI divergence from midpoint
            ])
            
            # MACD signal
            macd = data[i].get('macd_line') or 0
            feature_vector.extend([
                1 if macd > 0 else 0,  # MACD bullish
                abs(macd) * 10000      # MACD strength
            ])
            
            features.append(feature_vector)
        
        # Define feature names
        if not feature_names:
            feature_names = [
                'open', 'high', 'low', 'close', 'log_volume',
                'price_change', 'hl_spread', 'oc_change',
                'sma_20', 'ema_20', 'rsi_norm', 'macd_scaled', 'bb_upper', 'bb_lower',
                'atr_scaled', 'stoch_k_norm', 'stoch_d_norm', 'williams_norm', 'cci_norm',
                'price_vs_sma', 'price_vs_ema', 'price_vs_bb_upper', 'price_vs_bb_lower',
                'momentum_5', 'acceleration', 'volatility',
                'rsi_overbought', 'rsi_oversold', 'rsi_divergence',
                'macd_bullish', 'macd_strength'
            ]
        
        logger.info(f"Created {len(features)} samples with {len(features[0])} enhanced features")
        return features, feature_names
    
    @staticmethod
    def normalize_features(X: List[List[float]]) -> List[List[float]]:
        """Normalize features to improve model performance."""
        if not X:
            return X
        
        n_features = len(X[0])
        
        # Calculate mean and std for each feature
        means = []
        stds = []
        
        for feature_idx in range(n_features):
            values = [sample[feature_idx] for sample in X]
            mean_val = statistics.mean(values)
            std_val = statistics.stdev(values) if len(values) > 1 else 1.0
            
            means.append(mean_val)
            stds.append(std_val if std_val > 0 else 1.0)
        
        # Normalize features
        normalized_X = []
        for sample in X:
            normalized_sample = []
            for i, value in enumerate(sample):
                normalized_value = (value - means[i]) / stds[i]
                normalized_sample.append(normalized_value)
            normalized_X.append(normalized_sample)
        
        return normalized_X


class EnhancedLogisticRegression:
    """Enhanced logistic regression with regularization and better optimization."""
    
    def __init__(self, learning_rate: float = 0.1, max_iterations: int = 2000, 
                 regularization: float = 0.01, tolerance: float = 1e-6):
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.regularization = regularization
        self.tolerance = tolerance
        self.weights = None
        self.bias = 0
        
    def sigmoid(self, z):
        """Sigmoid with improved numerical stability."""
        z = max(-500, min(500, z))
        if z < 0:
            exp_z = math.exp(z)
            return exp_z / (1 + exp_z)
        else:
            return 1 / (1 + math.exp(-z))
    
    def fit(self, X: List[List[float]], y: List[int]):
        """Train with regularization and convergence detection."""
        n_samples = len(X)
        n_features = len(X[0]) if X else 0
        
        # Initialize weights with small random values
        self.weights = [(hash(str(i)) % 1000 - 500) / 10000 for i in range(n_features)]
        self.bias = 0.0
        
        prev_loss = float('inf')
        
        for iteration in range(self.max_iterations):
            # Forward pass
            y_pred = []
            for i in range(n_samples):
                z = sum(self.weights[j] * X[i][j] for j in range(n_features)) + self.bias
                y_pred.append(self.sigmoid(z))
            
            # Compute gradients with L2 regularization
            dw = [0.0] * n_features
            db = 0.0
            
            for i in range(n_samples):
                error = y_pred[i] - y[i]
                for j in range(n_features):
                    dw[j] += error * X[i][j]
                db += error
            
            # Add regularization
            for j in range(n_features):
                dw[j] = dw[j] / n_samples + self.regularization * self.weights[j]
            db = db / n_samples
            
            # Update parameters
            for j in range(n_features):
                self.weights[j] -= self.learning_rate * dw[j]
            self.bias -= self.learning_rate * db
            
            # Check convergence
            if iteration % 100 == 0:
                loss = self._compute_loss(X, y)
                logger.debug(f"LR Iteration {iteration}, Loss: {loss:.6f}")
                
                if abs(prev_loss - loss) < self.tolerance:
                    logger.info(f"LR converged at iteration {iteration}")
                    break
                prev_loss = loss
    
    def predict_proba(self, X: List[List[float]]) -> List[float]:
        """Predict probabilities."""
        if self.weights is None:
            raise ValueError("Model must be trained first")
            
        probabilities = []
        for sample in X:
            z = sum(self.weights[j] * sample[j] for j in range(len(sample))) + self.bias
            probabilities.append(self.sigmoid(z))
        return probabilities
    
    def predict(self, X: List[List[float]]) -> List[int]:
        """Predict with optimized threshold."""
        probabilities = self.predict_proba(X)
        # Use dynamic threshold based on class distribution
        threshold = 0.48  # Slightly lower threshold for better recall
        return [1 if prob >= threshold else 0 for prob in probabilities]
    
    def _compute_loss(self, X: List[List[float]], y: List[int]) -> float:
        """Compute regularized logistic loss."""
        total_loss = 0
        for i, sample in enumerate(X):
            z = sum(self.weights[j] * sample[j] for j in range(len(sample))) + self.bias
            prob = self.sigmoid(z)
            prob = max(1e-15, min(1-1e-15, prob))
            total_loss += -(y[i] * math.log(prob) + (1 - y[i]) * math.log(1 - prob))
        
        # Add regularization term
        reg_term = self.regularization * sum(w * w for w in self.weights) / 2
        return (total_loss / len(X)) + reg_term


class EnhancedRandomForest:
    """Enhanced random forest with better tree construction and ensemble methods."""
    
    def __init__(self, n_estimators: int = 50, max_depth: int = 8, 
                 min_samples_split: int = 10, feature_fraction: float = 0.7):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.feature_fraction = feature_fraction
        self.trees = []
        
    def fit(self, X: List[List[float]], y: List[int]):
        """Train enhanced random forest."""
        n_samples = len(X)
        n_features = len(X[0]) if X else 0
        n_selected_features = max(1, int(n_features * self.feature_fraction))
        
        self.trees = []
        for i in range(self.n_estimators):
            # Bootstrap sampling with replacement
            bootstrap_indices = [int(abs(hash(f"{i}_{j}")) % n_samples) for j in range(n_samples)]
            bootstrap_X = [X[idx] for idx in bootstrap_indices]
            bootstrap_y = [y[idx] for idx in bootstrap_indices]
            
            # Random feature selection
            feature_indices = sorted(set(int(abs(hash(f"{i}_{j}")) % n_features) 
                                        for j in range(n_selected_features)))[:n_selected_features]
            
            # Create decision tree
            tree = self._build_tree(bootstrap_X, bootstrap_y, feature_indices, 0)
            self.trees.append((tree, feature_indices))
    
    def _build_tree(self, X: List[List[float]], y: List[int], 
                   feature_indices: List[int], depth: int) -> Dict:
        """Build decision tree recursively."""
        # Base cases
        if (depth >= self.max_depth or 
            len(X) < self.min_samples_split or 
            len(set(y)) == 1):
            return {'prediction': self._majority_class(y)}
        
        best_feature = None
        best_threshold = None
        best_gini = 1.0
        best_split = None
        
        # Find best split
        for feature_idx in feature_indices:
            values = sorted(set(sample[feature_idx] for sample in X))
            if len(values) < 2:
                continue
            
            # Try multiple thresholds
            for i in range(0, len(values) - 1, max(1, len(values) // 10)):
                threshold = (values[i] + values[i + 1]) / 2
                
                left_indices = [j for j, sample in enumerate(X) 
                              if sample[feature_idx] <= threshold]
                right_indices = [j for j, sample in enumerate(X) 
                               if sample[feature_idx] > threshold]
                
                if not left_indices or not right_indices:
                    continue
                
                left_y = [y[j] for j in left_indices]
                right_y = [y[j] for j in right_indices]
                
                # Calculate weighted gini impurity
                total = len(left_y) + len(right_y)
                weighted_gini = (len(left_y) / total * self._gini_impurity(left_y) +
                               len(right_y) / total * self._gini_impurity(right_y))
                
                if weighted_gini < best_gini:
                    best_gini = weighted_gini
                    best_feature = feature_idx
                    best_threshold = threshold
                    best_split = (left_indices, right_indices)
        
        if best_split is None:
            return {'prediction': self._majority_class(y)}
        
        left_indices, right_indices = best_split
        left_X = [X[i] for i in left_indices]
        left_y = [y[i] for i in left_indices]
        right_X = [X[i] for i in right_indices]
        right_y = [y[i] for i in right_indices]
        
        return {
            'feature': best_feature,
            'threshold': best_threshold,
            'left': self._build_tree(left_X, left_y, feature_indices, depth + 1),
            'right': self._build_tree(right_X, right_y, feature_indices, depth + 1)
        }
    
    def _predict_tree(self, sample: List[float], tree: Dict, feature_indices: List[int]) -> float:
        """Predict using a single tree."""
        if 'prediction' in tree:
            return tree['prediction']
        
        if sample[tree['feature']] <= tree['threshold']:
            return self._predict_tree(sample, tree['left'], feature_indices)
        else:
            return self._predict_tree(sample, tree['right'], feature_indices)
    
    def predict_proba(self, X: List[List[float]]) -> List[float]:
        """Predict probabilities using ensemble voting."""
        if not self.trees:
            raise ValueError("Model must be trained first")
        
        probabilities = []
        for sample in X:
            votes = []
            for tree, feature_indices in self.trees:
                vote = self._predict_tree(sample, tree, feature_indices)
                votes.append(vote)
            
            probability = sum(votes) / len(votes)
            probabilities.append(probability)
        
        return probabilities
    
    def predict(self, X: List[List[float]]) -> List[int]:
        """Predict classes."""
        probabilities = self.predict_proba(X)
        return [1 if prob >= 0.5 else 0 for prob in probabilities]
    
    def _gini_impurity(self, y: List[int]) -> float:
        """Calculate Gini impurity."""
        if not y:
            return 0
        
        counts = {}
        for label in y:
            counts[label] = counts.get(label, 0) + 1
        
        impurity = 1.0
        total = len(y)
        for count in counts.values():
            prob = count / total
            impurity -= prob * prob
        
        return impurity
    
    def _majority_class(self, y: List[int]) -> float:
        """Get the majority class as probability."""
        if not y:
            return 0.5
        
        counts = {0: 0, 1: 0}
        for label in y:
            counts[label] = counts.get(label, 0) + 1
        
        return counts[1] / len(y)


class EnhancedModelPipeline:
    """Enhanced model pipeline with better feature engineering and evaluation."""
    
    def __init__(self, data_path: str = "data/trading_bot.db"):
        self.data_path = data_path
        self.models = {}
        self.feature_names = []
        self.feature_engineer = FeatureEngineer()
        
    def load_and_prepare_data(self) -> Tuple[List[List[float]], List[int], List[str]]:
        """Load and prepare enhanced data."""
        logger.info("Loading and preparing enhanced data...")
        
        if not os.path.exists(self.data_path):
            logger.error(f"Database not found at {self.data_path}")
            return [], [], []
        
        try:
            conn = sqlite3.connect(self.data_path)
            cursor = conn.cursor()
            
            # Load comprehensive data
            query = """
            SELECT 
                p.timestamp, p.open, p.high, p.low, p.close, p.volume,
                ti.sma_20, ti.ema_20, ti.rsi_14, ti.macd_line, 
                ti.bb_upper, ti.bb_lower, ti.atr_14, 
                ti.stoch_k, ti.stoch_d, ti.williams_r, ti.cci_20
            FROM price_data p
            LEFT JOIN technical_indicators ti ON p.id = ti.price_data_id
            WHERE p.symbol = 'EURUSD=X'
            ORDER BY p.timestamp
            """
            
            cursor.execute(query)
            rows = cursor.fetchall()
            conn.close()
            
            if len(rows) < 50:
                logger.error("Insufficient data for training")
                return [], [], []
            
            # Convert to dict format for feature engineering
            data_dicts = []
            for row in rows:
                record = {
                    'timestamp': row[0],
                    'open': float(row[1]) if row[1] else 0,
                    'high': float(row[2]) if row[2] else 0,
                    'low': float(row[3]) if row[3] else 0,
                    'close': float(row[4]) if row[4] else 0,
                    'volume': int(row[5]) if row[5] else 1,
                    'sma_20': float(row[6]) if row[6] else None,
                    'ema_20': float(row[7]) if row[7] else None,
                    'rsi_14': float(row[8]) if row[8] else None,
                    'macd_line': float(row[9]) if row[9] else None,
                    'bb_upper': float(row[10]) if row[10] else None,
                    'bb_lower': float(row[11]) if row[11] else None,
                    'atr_14': float(row[12]) if row[12] else None,
                    'stoch_k': float(row[13]) if row[13] else None,
                    'stoch_d': float(row[14]) if row[14] else None,
                    'williams_r': float(row[15]) if row[15] else None,
                    'cci_20': float(row[16]) if row[16] else None,
                }
                data_dicts.append(record)
            
            # Create enhanced features
            X, self.feature_names = self.feature_engineer.create_enhanced_features(data_dicts)
            
            # Create improved labels (multi-period prediction)
            y = []
            timestamps = []
            
            for i in range(len(data_dicts) - 5):  # Look ahead 5 periods
                current_close = data_dicts[i]['close']
                future_close = data_dicts[i + 5]['close']
                
                # Label: 1 if price increases by more than 0.01% in next 5 periods
                price_change = (future_close - current_close) / current_close
                label = 1 if price_change > 0.0001 else 0  # 0.01% threshold
                
                y.append(label)
                timestamps.append(data_dicts[i]['timestamp'])
            
            # Trim features to match labels
            X = X[:len(y)]
            
            # Normalize features
            X = self.feature_engineer.normalize_features(X)
            
            logger.info(f"Enhanced data prepared: {len(X)} samples, {len(X[0])} features")
            logger.info(f"Class distribution: {sum(y)} positive ({sum(y)/len(y)*100:.1f}%), {len(y) - sum(y)} negative")
            
            return X, y, timestamps
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return [], [], []
    
    def train_enhanced_models(self, X_train: List[List[float]], y_train: List[int]):
        """Train enhanced models."""
        logger.info("Training enhanced baseline models...")
        
        # Enhanced Logistic Regression
        logger.info("Training Enhanced Logistic Regression...")
        lr_model = EnhancedLogisticRegression(
            learning_rate=0.01, 
            max_iterations=1000,
            regularization=0.001
        )
        lr_model.fit(X_train, y_train)
        self.models['enhanced_logistic_regression'] = lr_model
        
        # Enhanced Random Forest
        logger.info("Training Enhanced Random Forest...")
        rf_model = EnhancedRandomForest(
            n_estimators=30,
            max_depth=6,
            min_samples_split=20,
            feature_fraction=0.8
        )
        rf_model.fit(X_train, y_train)
        self.models['enhanced_random_forest'] = rf_model
        
        logger.info("Enhanced model training completed!")
    
    def evaluate_enhanced_models(self, X_test: List[List[float]], y_test: List[int]) -> List[EnhancedModelResult]:
        """Evaluate models with enhanced metrics."""
        logger.info("Evaluating enhanced models...")
        results = []
        
        for model_name, model in self.models.items():
            logger.info(f"Evaluating {model_name}...")
            
            predictions = model.predict(X_test)
            probabilities = model.predict_proba(X_test)
            
            # Calculate standard metrics
            accuracy = self._calculate_accuracy(y_test, predictions)
            precision = self._calculate_precision(y_test, predictions)
            recall = self._calculate_recall(y_test, predictions)
            f1 = self._calculate_f1_score(precision, recall)
            
            # Calculate trading-specific metrics
            directional_accuracy = self._calculate_directional_accuracy(y_test, predictions)
            profitable_trade_rate = self._calculate_profitable_trade_rate(y_test, predictions, probabilities)
            
            result = EnhancedModelResult(
                model_name=model_name,
                predictions=predictions,
                probabilities=probabilities,
                actual=y_test,
                accuracy=accuracy,
                precision=precision,
                recall=recall,
                f1_score=f1,
                directional_accuracy=directional_accuracy,
                profitable_trade_rate=profitable_trade_rate,
                metadata={
                    'n_test_samples': len(y_test),
                    'feature_names': self.feature_names[:10],  # First 10 features
                    'model_type': type(model).__name__,
                    'class_distribution': f"{sum(y_test)}/{len(y_test) - sum(y_test)}"
                }
            )
            
            results.append(result)
            
            logger.info(f"{model_name} Results:")
            logger.info(f"  Accuracy: {accuracy:.4f}")
            logger.info(f"  Precision: {precision:.4f}")
            logger.info(f"  Recall: {recall:.4f}")
            logger.info(f"  F1-Score: {f1:.4f}")
            logger.info(f"  Directional Accuracy: {directional_accuracy:.4f}")
            logger.info(f"  Profitable Trade Rate: {profitable_trade_rate:.4f}")
        
        return results
    
    def _calculate_accuracy(self, y_true: List[int], y_pred: List[int]) -> float:
        """Calculate accuracy."""
        if not y_true:
            return 0.0
        correct = sum(1 for true, pred in zip(y_true, y_pred) if true == pred)
        return correct / len(y_true)
    
    def _calculate_precision(self, y_true: List[int], y_pred: List[int]) -> float:
        """Calculate precision."""
        true_positives = sum(1 for true, pred in zip(y_true, y_pred) if true == 1 and pred == 1)
        predicted_positives = sum(y_pred)
        return true_positives / predicted_positives if predicted_positives > 0 else 0.0
    
    def _calculate_recall(self, y_true: List[int], y_pred: List[int]) -> float:
        """Calculate recall."""
        true_positives = sum(1 for true, pred in zip(y_true, y_pred) if true == 1 and pred == 1)
        actual_positives = sum(y_true)
        return true_positives / actual_positives if actual_positives > 0 else 0.0
    
    def _calculate_f1_score(self, precision: float, recall: float) -> float:
        """Calculate F1 score."""
        return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    def _calculate_directional_accuracy(self, y_true: List[int], y_pred: List[int]) -> float:
        """Calculate directional accuracy (forex-specific)."""
        return self._calculate_accuracy(y_true, y_pred)
    
    def _calculate_profitable_trade_rate(self, y_true: List[int], y_pred: List[int], 
                                       probabilities: List[float]) -> float:
        """Calculate rate of profitable trades based on confidence."""
        if not probabilities:
            return 0.0
        
        # Only consider high-confidence predictions
        high_confidence_trades = [(true, pred) for true, pred, prob in 
                                zip(y_true, y_pred, probabilities) 
                                if prob > 0.6 or prob < 0.4]
        
        if not high_confidence_trades:
            return 0.0
        
        correct_trades = sum(1 for true, pred in high_confidence_trades if true == pred)
        return correct_trades / len(high_confidence_trades)
    
    def save_enhanced_results(self, results: List[EnhancedModelResult], 
                            output_path: str = "data/enhanced_baseline_results.json"):
        """Save enhanced results."""
        results_data = {
            'timestamp': datetime.now().isoformat(),
            'stage': 'Stage 3 - Enhanced Baseline Models',
            'target_criteria': {
                'accuracy': 0.60,
                'precision': 0.55,
                'directional_accuracy': 0.60
            },
            'models': {}
        }
        
        for result in results:
            meets_criteria = (result.accuracy >= 0.60 and 
                            result.directional_accuracy >= 0.60)
            
            results_data['models'][result.model_name] = {
                'accuracy': result.accuracy,
                'precision': result.precision,
                'recall': result.recall,
                'f1_score': result.f1_score,
                'directional_accuracy': result.directional_accuracy,
                'profitable_trade_rate': result.profitable_trade_rate,
                'meets_baseline_criteria': meets_criteria,
                'n_samples': len(result.predictions),
                'metadata': result.metadata
            }
        
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        logger.info(f"Enhanced results saved to {output_path}")


def main():
    """Main function for enhanced baseline model training."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger.info("Starting enhanced baseline model training pipeline...")
    
    # Initialize pipeline
    pipeline = EnhancedModelPipeline()
    
    # Load enhanced data
    X, y, timestamps = pipeline.load_and_prepare_data()
    
    if not X or not y:
        logger.error("No enhanced data available for training")
        return
    
    logger.info(f"Enhanced data loaded: {len(X)} samples, {len(X[0])} features")
    
    # Split data chronologically
    split_idx = int(len(X) * 0.7)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    logger.info(f"Data split: {len(X_train)} train, {len(X_test)} test samples")
    
    # Train models
    pipeline.train_enhanced_models(X_train, y_train)
    
    # Evaluate models
    results = pipeline.evaluate_enhanced_models(X_test, y_test)
    
    # Save results
    pipeline.save_enhanced_results(results)
    
    # Print comprehensive summary
    print("\n" + "="*70)
    print("ENHANCED BASELINE MODEL RESULTS - STAGE 3")
    print("="*70)
    
    best_accuracy = 0
    best_model = None
    
    for result in results:
        meets_baseline = result.accuracy >= 0.60
        meets_directional = result.directional_accuracy >= 0.60
        overall_meets = meets_baseline and meets_directional
        
        print(f"\n{result.model_name.upper().replace('_', ' ')}:")
        print(f"  Accuracy:             {result.accuracy:.4f} ({'✅' if meets_baseline else '❌'} Target: ≥0.60)")
        print(f"  Directional Accuracy: {result.directional_accuracy:.4f} ({'✅' if meets_directional else '❌'} Target: ≥0.60)")
        print(f"  Precision:            {result.precision:.4f}")
        print(f"  Recall:               {result.recall:.4f}")
        print(f"  F1-Score:             {result.f1_score:.4f}")
        print(f"  Profitable Trade Rate: {result.profitable_trade_rate:.4f}")
        print(f"  Baseline Status:      {'✅ MEETS CRITERIA' if overall_meets else '❌ BELOW BASELINE'}")
        
        if result.accuracy > best_accuracy:
            best_accuracy = result.accuracy
            best_model = result.model_name
    
    # Overall Stage 3 Assessment
    stage_3_success = best_accuracy >= 0.60
    
    print(f"\n{'='*70}")
    print("STAGE 3 BASELINE MODEL ASSESSMENT:")
    print(f"Best Model: {best_model}")
    print(f"Best Accuracy: {best_accuracy:.4f}")
    print(f"Baseline Criteria (≥60%): {'✅ MET' if stage_3_success else '❌ NOT MET'}")
    
    if stage_3_success:
        print("\n🎯 STAGE 3 BASELINE COMPLETED SUCCESSFULLY!")
        print("✅ Ready to advance to Stage 3.2: Advanced Neural Network Models (LSTM, GRU)")
        print("   Target: >75% accuracy with time-series deep learning")
    else:
        print("\n⚠️  BASELINE MODELS NEED FURTHER IMPROVEMENT")
        print("   Consider: More data, feature engineering, hyperparameter tuning")
    
    print("="*70)
    
    logger.info("Enhanced baseline model pipeline completed!")


if __name__ == "__main__":
    main()