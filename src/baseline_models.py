"""
Baseline Models for Forex Trading Bot

This module implements baseline predictive models for forex direction prediction
using logistic regression and simple tree-based approaches with minimal dependencies.

The models predict price direction (up/down) for the next period based on
technical indicators and engineered features.
"""

import sqlite3
import json
import csv
import math
import statistics
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
import os

logger = logging.getLogger(__name__)


@dataclass
class ModelResult:
    """Result container for model predictions and performance."""
    model_name: str
    predictions: List[int]
    probabilities: List[float]
    actual: List[int]
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    metadata: Dict[str, Any]


class SimpleLogisticRegression:
    """
    Simple logistic regression implementation using gradient descent.
    Minimal dependency implementation for binary classification.
    """
    
    def __init__(self, learning_rate: float = 0.01, max_iterations: int = 1000):
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.weights = None
        self.bias = 0
        
    def sigmoid(self, z):
        """Sigmoid activation function with overflow protection."""
        z = max(-250, min(250, z))  # Prevent overflow
        return 1 / (1 + math.exp(-z))
    
    def fit(self, X: List[List[float]], y: List[int]):
        """Train the logistic regression model."""
        n_samples = len(X)
        n_features = len(X[0]) if X else 0
        
        # Initialize weights
        self.weights = [0.0] * n_features
        self.bias = 0.0
        
        for iteration in range(self.max_iterations):
            # Forward pass
            y_pred = []
            for i in range(n_samples):
                z = sum(self.weights[j] * X[i][j] for j in range(n_features)) + self.bias
                y_pred.append(self.sigmoid(z))
            
            # Compute gradients
            dw = [0.0] * n_features
            db = 0.0
            
            for i in range(n_samples):
                error = y_pred[i] - y[i]
                for j in range(n_features):
                    dw[j] += error * X[i][j]
                db += error
            
            # Update parameters
            for j in range(n_features):
                self.weights[j] -= self.learning_rate * dw[j] / n_samples
            self.bias -= self.learning_rate * db / n_samples
            
            # Check convergence every 100 iterations
            if iteration % 100 == 0:
                loss = self._compute_loss(X, y)
                logger.debug(f"Iteration {iteration}, Loss: {loss:.4f}")
                
    def predict_proba(self, X: List[List[float]]) -> List[float]:
        """Predict probabilities for the positive class."""
        if self.weights is None:
            raise ValueError("Model must be trained first")
            
        probabilities = []
        for sample in X:
            z = sum(self.weights[j] * sample[j] for j in range(len(sample))) + self.bias
            probabilities.append(self.sigmoid(z))
        return probabilities
    
    def predict(self, X: List[List[float]]) -> List[int]:
        """Predict binary classes."""
        probabilities = self.predict_proba(X)
        return [1 if prob >= 0.5 else 0 for prob in probabilities]
    
    def _compute_loss(self, X: List[List[float]], y: List[int]) -> float:
        """Compute logistic loss for monitoring convergence."""
        total_loss = 0
        for i, sample in enumerate(X):
            z = sum(self.weights[j] * sample[j] for j in range(len(sample))) + self.bias
            prob = self.sigmoid(z)
            # Avoid log(0)
            prob = max(1e-15, min(1-1e-15, prob))
            total_loss += -(y[i] * math.log(prob) + (1 - y[i]) * math.log(1 - prob))
        return total_loss / len(X)


class SimpleRandomForest:
    """
    Simple random forest implementation using multiple decision stumps.
    Minimal dependency implementation for ensemble learning.
    """
    
    def __init__(self, n_estimators: int = 10, max_depth: int = 5):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.trees = []
        
    def fit(self, X: List[List[float]], y: List[int]):
        """Train the random forest."""
        n_samples = len(X)
        n_features = len(X[0]) if X else 0
        
        self.trees = []
        for i in range(self.n_estimators):
            # Bootstrap sampling
            bootstrap_X = []
            bootstrap_y = []
            for _ in range(n_samples):
                idx = int(hash(f"{i}_{_}") % n_samples)  # Simple pseudo-random
                bootstrap_X.append(X[idx])
                bootstrap_y.append(y[idx])
            
            # Create simple decision stump
            tree = self._create_decision_stump(bootstrap_X, bootstrap_y, n_features)
            self.trees.append(tree)
    
    def _create_decision_stump(self, X: List[List[float]], y: List[int], n_features: int) -> Dict:
        """Create a simple decision stump (depth-1 tree)."""
        best_feature = 0
        best_threshold = 0
        best_gini = 1.0
        
        # Simple feature selection (use modulo for pseudo-randomness)
        selected_features = [i for i in range(0, n_features, max(1, n_features // 5))]
        
        for feature_idx in selected_features:
            # Get unique values for threshold candidates
            values = sorted(set(sample[feature_idx] for sample in X))
            if len(values) < 2:
                continue
                
            for i in range(len(values) - 1):
                threshold = (values[i] + values[i + 1]) / 2
                
                # Split data
                left_y = [y[j] for j, sample in enumerate(X) if sample[feature_idx] <= threshold]
                right_y = [y[j] for j, sample in enumerate(X) if sample[feature_idx] > threshold]
                
                if not left_y or not right_y:
                    continue
                
                # Calculate weighted gini impurity
                total = len(left_y) + len(right_y)
                left_gini = self._gini_impurity(left_y)
                right_gini = self._gini_impurity(right_y)
                weighted_gini = (len(left_y) / total) * left_gini + (len(right_y) / total) * right_gini
                
                if weighted_gini < best_gini:
                    best_gini = weighted_gini
                    best_feature = feature_idx
                    best_threshold = threshold
        
        # Create leaf predictions
        left_samples = [y[j] for j, sample in enumerate(X) if sample[best_feature] <= best_threshold]
        right_samples = [y[j] for j, sample in enumerate(X) if sample[best_feature] > best_threshold]
        
        left_prediction = self._majority_class(left_samples) if left_samples else 0
        right_prediction = self._majority_class(right_samples) if right_samples else 0
        
        return {
            'feature': best_feature,
            'threshold': best_threshold,
            'left_prediction': left_prediction,
            'right_prediction': right_prediction
        }
    
    def _gini_impurity(self, y: List[int]) -> float:
        """Calculate Gini impurity."""
        if not y:
            return 0
        
        counts = {}
        for label in y:
            counts[label] = counts.get(label, 0) + 1
        
        impurity = 1.0
        for count in counts.values():
            prob = count / len(y)
            impurity -= prob * prob
        
        return impurity
    
    def _majority_class(self, y: List[int]) -> int:
        """Get the majority class."""
        if not y:
            return 0
        
        counts = {}
        for label in y:
            counts[label] = counts.get(label, 0) + 1
        
        return max(counts.keys(), key=lambda k: counts[k])
    
    def predict_proba(self, X: List[List[float]]) -> List[float]:
        """Predict probabilities by averaging tree predictions."""
        if not self.trees:
            raise ValueError("Model must be trained first")
        
        probabilities = []
        for sample in X:
            votes = []
            for tree in self.trees:
                if sample[tree['feature']] <= tree['threshold']:
                    votes.append(tree['left_prediction'])
                else:
                    votes.append(tree['right_prediction'])
            
            # Convert votes to probability
            prob = sum(votes) / len(votes)
            probabilities.append(prob)
        
        return probabilities
    
    def predict(self, X: List[List[float]]) -> List[int]:
        """Predict classes by majority voting."""
        probabilities = self.predict_proba(X)
        return [1 if prob >= 0.5 else 0 for prob in probabilities]


class BaselineModelPipeline:
    """
    Complete pipeline for baseline model training and evaluation.
    Handles data loading, feature preparation, model training, and evaluation.
    """
    
    def __init__(self, data_path: str = "data/trading_bot.db"):
        self.data_path = data_path
        self.models = {}
        self.feature_names = []
        
    def load_data(self) -> Tuple[List[List[float]], List[int], List[str]]:
        """
        Load and prepare data for modeling.
        Returns features, labels, and timestamps.
        """
        logger.info("Loading data for model training...")
        
        if not os.path.exists(self.data_path):
            logger.error(f"Database not found at {self.data_path}")
            # Return dummy data for testing
            return self._generate_dummy_data()
        
        try:
            conn = sqlite3.connect(self.data_path)
            cursor = conn.cursor()
            
            # Load price data with technical indicators
            query = """
            SELECT 
                p.timestamp, p.open, p.high, p.low, p.close, p.volume,
                ti.sma_20, ti.ema_20, ti.rsi_14, ti.macd_line, ti.bb_upper, ti.bb_lower,
                ti.atr_14, ti.stoch_k, ti.stoch_d, ti.williams_r, ti.cci_20
            FROM price_data p
            LEFT JOIN technical_indicators ti ON p.id = ti.price_data_id
            WHERE p.symbol = 'EURUSD=X'
            ORDER BY p.timestamp
            LIMIT 10000
            """
            
            cursor.execute(query)
            rows = cursor.fetchall()
            conn.close()
            
            if not rows:
                logger.warning("No data found in database, generating dummy data")
                return self._generate_dummy_data()
            
            return self._prepare_features(rows)
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return self._generate_dummy_data()
    
    def _generate_dummy_data(self) -> Tuple[List[List[float]], List[int], List[str]]:
        """Generate dummy data for testing when real data is not available."""
        logger.info("Generating dummy forex data for baseline model testing...")
        
        n_samples = 1000
        n_features = 10
        
        # Generate realistic forex-like features
        features = []
        labels = []
        timestamps = []
        
        base_price = 1.1000  # EURUSD base price
        
        for i in range(n_samples):
            # Simulate price movement (random walk with slight trend)
            price_change = (hash(str(i)) % 200 - 100) / 10000  # -0.01 to 0.01
            current_price = base_price + (i * 0.00001) + price_change
            
            # Generate technical indicator-like features
            feature_vector = [
                current_price,  # Close price
                current_price * (1 + (hash(str(i+1)) % 100 - 50) / 50000),  # SMA-like
                current_price * (1 + (hash(str(i+2)) % 100 - 50) / 50000),  # EMA-like
                50 + (hash(str(i+3)) % 100),  # RSI-like (0-100)
                (hash(str(i+4)) % 200 - 100) / 1000,  # MACD-like
                current_price * 1.02,  # BB upper
                current_price * 0.98,  # BB lower
                abs(hash(str(i+5)) % 100) / 1000,  # ATR-like
                hash(str(i+6)) % 100,  # Stoch K
                hash(str(i+7)) % 100,  # Stoch D
                -100 + (hash(str(i+8)) % 200)  # Williams %R-like
            ]
            
            features.append(feature_vector)
            
            # Generate label (price direction): 1 if next price higher, 0 if lower
            next_price_change = (hash(str(i+100)) % 200 - 100) / 10000
            label = 1 if next_price_change > 0 else 0
            labels.append(label)
            
            # Generate timestamp
            timestamp = f"2024-01-01 {i:02d}:00:00"
            timestamps.append(timestamp)
        
        self.feature_names = [
            "close_price", "sma_20", "ema_20", "rsi_14", "macd_line",
            "bb_upper", "bb_lower", "atr_14", "stoch_k", "stoch_d", "williams_r"
        ]
        
        logger.info(f"Generated {len(features)} samples with {len(features[0])} features")
        return features, labels, timestamps
    
    def _prepare_features(self, rows: List[Tuple]) -> Tuple[List[List[float]], List[int], List[str]]:
        """Prepare features from database rows."""
        features = []
        labels = []
        timestamps = []
        
        self.feature_names = [
            "open", "high", "low", "close", "volume", "sma_20", "ema_20", "rsi_14",
            "macd_line", "bb_upper", "bb_lower", "atr_14", "stoch_k", "stoch_d",
            "williams_r", "cci_20"
        ]
        
        for i in range(len(rows) - 1):  # -1 because we need next price for label
            current_row = rows[i]
            next_row = rows[i + 1]
            
            # Extract features (skip timestamp and handle nulls)
            feature_vector = []
            for j in range(1, len(current_row)):
                value = current_row[j] if current_row[j] is not None else 0.0
                feature_vector.append(float(value))
            
            features.append(feature_vector)
            
            # Create label: 1 if next close > current close, 0 otherwise
            current_close = current_row[4] if current_row[4] is not None else 0
            next_close = next_row[4] if next_row[4] is not None else 0
            label = 1 if next_close > current_close else 0
            labels.append(label)
            
            timestamps.append(current_row[0])
        
        logger.info(f"Prepared {len(features)} samples with {len(features[0])} features")
        return features, labels, timestamps
    
    def train_test_split(self, X: List[List[float]], y: List[int], 
                        test_size: float = 0.2) -> Tuple[List[List[float]], List[List[float]], 
                                                        List[int], List[int]]:
        """Split data chronologically (important for time series)."""
        split_idx = int(len(X) * (1 - test_size))
        return X[:split_idx], X[split_idx:], y[:split_idx], y[split_idx:]
    
    def train_models(self, X_train: List[List[float]], y_train: List[int]):
        """Train all baseline models."""
        logger.info("Training baseline models...")
        
        # Train Logistic Regression
        logger.info("Training Logistic Regression...")
        lr_model = SimpleLogisticRegression(learning_rate=0.01, max_iterations=500)
        lr_model.fit(X_train, y_train)
        self.models['logistic_regression'] = lr_model
        
        # Train Random Forest
        logger.info("Training Random Forest...")
        rf_model = SimpleRandomForest(n_estimators=20, max_depth=5)
        rf_model.fit(X_train, y_train)
        self.models['random_forest'] = rf_model
        
        logger.info("Model training completed!")
    
    def evaluate_models(self, X_test: List[List[float]], y_test: List[int]) -> List[ModelResult]:
        """Evaluate all trained models."""
        logger.info("Evaluating models...")
        results = []
        
        for model_name, model in self.models.items():
            logger.info(f"Evaluating {model_name}...")
            
            # Make predictions
            predictions = model.predict(X_test)
            probabilities = model.predict_proba(X_test)
            
            # Calculate metrics
            accuracy = self._calculate_accuracy(y_test, predictions)
            precision = self._calculate_precision(y_test, predictions)
            recall = self._calculate_recall(y_test, predictions)
            f1 = self._calculate_f1_score(precision, recall)
            
            result = ModelResult(
                model_name=model_name,
                predictions=predictions,
                probabilities=probabilities,
                actual=y_test,
                accuracy=accuracy,
                precision=precision,
                recall=recall,
                f1_score=f1,
                metadata={
                    'n_test_samples': len(y_test),
                    'feature_names': self.feature_names,
                    'model_type': type(model).__name__
                }
            )
            
            results.append(result)
            
            logger.info(f"{model_name} Results:")
            logger.info(f"  Accuracy: {accuracy:.4f}")
            logger.info(f"  Precision: {precision:.4f}")
            logger.info(f"  Recall: {recall:.4f}")
            logger.info(f"  F1-Score: {f1:.4f}")
        
        return results
    
    def _calculate_accuracy(self, y_true: List[int], y_pred: List[int]) -> float:
        """Calculate accuracy."""
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
    
    def save_results(self, results: List[ModelResult], output_path: str = "data/baseline_model_results.json"):
        """Save model results to file."""
        results_data = {
            'timestamp': datetime.now().isoformat(),
            'models': {}
        }
        
        for result in results:
            results_data['models'][result.model_name] = {
                'accuracy': result.accuracy,
                'precision': result.precision,
                'recall': result.recall,
                'f1_score': result.f1_score,
                'n_samples': len(result.predictions),
                'metadata': result.metadata
            }
        
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        logger.info(f"Results saved to {output_path}")


def main():
    """Main function to run baseline model training and evaluation."""
    import sys
    import os
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger.info("Starting baseline model training pipeline...")
    
    # Initialize pipeline
    pipeline = BaselineModelPipeline()
    
    # Load data
    X, y, timestamps = pipeline.load_data()
    
    if not X or not y:
        logger.error("No data available for training")
        return
    
    logger.info(f"Data loaded: {len(X)} samples, {len(X[0])} features")
    logger.info(f"Class distribution: {sum(y)} positive, {len(y) - sum(y)} negative")
    
    # Split data
    X_train, X_test, y_train, y_test = pipeline.train_test_split(X, y, test_size=0.3)
    logger.info(f"Data split: {len(X_train)} train, {len(X_test)} test samples")
    
    # Train models
    pipeline.train_models(X_train, y_train)
    
    # Evaluate models
    results = pipeline.evaluate_models(X_test, y_test)
    
    # Save results
    pipeline.save_results(results)
    
    # Print summary
    print("\n" + "="*60)
    print("BASELINE MODEL RESULTS SUMMARY")
    print("="*60)
    
    for result in results:
        print(f"\n{result.model_name.upper()}:")
        print(f"  Accuracy:  {result.accuracy:.4f} ({'✅' if result.accuracy > 0.6 else '❌'} Target: >0.60)")
        print(f"  Precision: {result.precision:.4f}")
        print(f"  Recall:    {result.recall:.4f}")
        print(f"  F1-Score:  {result.f1_score:.4f}")
        
        # Check if meets baseline criteria
        meets_criteria = result.accuracy > 0.6
        print(f"  Status:    {'✅ MEETS BASELINE CRITERIA' if meets_criteria else '❌ BELOW BASELINE'}")
    
    # Overall assessment
    best_accuracy = max(result.accuracy for result in results)
    overall_success = best_accuracy > 0.6
    
    print(f"\n{'='*60}")
    print("STAGE 3 BASELINE ASSESSMENT:")
    print(f"Best Accuracy: {best_accuracy:.4f}")
    print(f"Baseline Criteria (>60%): {'✅ MET' if overall_success else '❌ NOT MET'}")
    
    if overall_success:
        print("🎯 READY TO ADVANCE TO ADVANCED MODELS (LSTM, GRU)")
    else:
        print("⚠️  NEED TO IMPROVE BASELINE BEFORE ADVANCING")
    
    print("="*60)
    
    logger.info("Baseline model pipeline completed!")


if __name__ == "__main__":
    main()