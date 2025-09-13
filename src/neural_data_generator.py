"""
Simple Neural Network Data Generator

Creates sample data specifically for neural network training without external dependencies.
"""

import sqlite3
import math
import random
import os
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)


class SimpleNeuralDataGenerator:
    """Generate realistic forex data for neural network training."""
    
    def __init__(self, db_path: str = "data/trading_bot.db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
    def create_database_schema(self):
        """Create database tables."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create price_data table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS price_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                open REAL NOT NULL,
                high REAL NOT NULL,
                low REAL NOT NULL,
                close REAL NOT NULL,
                volume INTEGER NOT NULL
            )
        ''')
        
        # Create technical_indicators table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS technical_indicators (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                price_data_id INTEGER,
                sma_20 REAL,
                ema_20 REAL,
                rsi_14 REAL,
                macd_line REAL,
                bb_upper REAL,
                bb_lower REAL,
                atr_14 REAL,
                stoch_k REAL,
                stoch_d REAL,
                williams_r REAL,
                cci_20 REAL,
                FOREIGN KEY (price_data_id) REFERENCES price_data (id)
            )
        ''')
        
        conn.commit()
        conn.close()
        logger.info("Database schema created successfully")
    
    def generate_neural_training_data(self, n_records: int = 5000) -> List[Dict]:
        """Generate realistic forex data optimized for neural networks."""
        logger.info(f"Generating {n_records} records for neural network training...")
        
        data = []
        base_price = 1.1000  # EURUSD starting price
        
        # Neural-network friendly parameters
        volatility = 0.0001  # Lower volatility for cleaner patterns
        trend_strength = 0.00002  # Subtle trend component
        mean_reversion = 0.95  # Strong mean reversion
        
        for i in range(n_records):
            # Time series generation with multiple patterns
            t = i / n_records * 4 * math.pi  # 4 complete cycles
            
            # Multiple time scale components
            trend = math.sin(t * 0.1) * trend_strength * 10  # Long-term trend
            cycle = math.sin(t) * volatility * 5              # Main cycle
            noise = random.gauss(0, volatility * 2)           # Random noise
            
            # Price evolution with mean reversion
            if i > 0:
                prev_price = data[-1]['close']
                reversion = (base_price - prev_price) * (1 - mean_reversion)
                price_change = trend + cycle + noise + reversion
                new_price = prev_price + price_change
            else:
                new_price = base_price
            
            # Ensure price stays realistic
            new_price = max(0.9000, min(1.3000, new_price))
            
            # Generate OHLC with realistic relationships
            volatility_factor = abs(random.gauss(1.0, 0.3))
            spread = volatility * volatility_factor * 2
            
            open_price = new_price + random.gauss(0, spread * 0.5)
            close_price = new_price
            
            high_price = max(open_price, close_price) + abs(random.gauss(0, spread))
            low_price = min(open_price, close_price) - abs(random.gauss(0, spread))
            
            volume = int(random.gauss(1000, 200))
            volume = max(100, volume)
            
            timestamp = (datetime(2024, 1, 1) + timedelta(minutes=i)).isoformat()
            
            record = {
                'timestamp': timestamp,
                'open': round(open_price, 5),
                'high': round(high_price, 5),
                'low': round(low_price, 5),
                'close': round(close_price, 5),
                'volume': volume
            }
            
            data.append(record)
        
        logger.info(f"Generated {len(data)} records with realistic patterns")
        return data
    
    def calculate_technical_indicators(self, data: List[Dict]) -> List[Dict]:
        """Calculate technical indicators for the data."""
        logger.info("Calculating technical indicators...")
        
        indicators = []
        
        for i in range(len(data)):
            # Simple Moving Average (20 periods)
            if i >= 19:
                sma_20 = sum(data[j]['close'] for j in range(i-19, i+1)) / 20
            else:
                sma_20 = data[i]['close']
            
            # Exponential Moving Average (20 periods)  
            if i == 0:
                ema_20 = data[i]['close']
            else:
                alpha = 2.0 / (20 + 1)
                ema_20 = alpha * data[i]['close'] + (1 - alpha) * indicators[i-1]['ema_20']
            
            # RSI (14 periods)
            if i >= 14:
                gains = []
                losses = []
                for j in range(i-13, i+1):
                    change = data[j]['close'] - data[j-1]['close'] if j > 0 else 0
                    if change > 0:
                        gains.append(change)
                        losses.append(0)
                    else:
                        gains.append(0)
                        losses.append(abs(change))
                
                avg_gain = sum(gains) / 14
                avg_loss = sum(losses) / 14
                
                if avg_loss > 0:
                    rs = avg_gain / avg_loss
                    rsi_14 = 100 - (100 / (1 + rs))
                else:
                    rsi_14 = 100
            else:
                rsi_14 = 50
            
            # MACD (12, 26, 9)
            if i >= 25:
                ema_12 = data[i]['close']  # Simplified
                ema_26 = indicators[i-25]['ema_20'] if i >= 25 else data[i]['close']
                macd_line = ema_12 - ema_26
            else:
                macd_line = 0
            
            # Bollinger Bands (20, 2)
            if i >= 19:
                close_prices = [data[j]['close'] for j in range(i-19, i+1)]
                bb_middle = sum(close_prices) / 20
                variance = sum((p - bb_middle) ** 2 for p in close_prices) / 20
                bb_std = math.sqrt(variance)
                bb_upper = bb_middle + 2 * bb_std
                bb_lower = bb_middle - 2 * bb_std
            else:
                bb_upper = data[i]['close'] * 1.01
                bb_lower = data[i]['close'] * 0.99
            
            # ATR (14 periods)
            if i >= 14:
                true_ranges = []
                for j in range(i-13, i+1):
                    if j > 0:
                        tr1 = data[j]['high'] - data[j]['low']
                        tr2 = abs(data[j]['high'] - data[j-1]['close'])
                        tr3 = abs(data[j]['low'] - data[j-1]['close'])
                        true_ranges.append(max(tr1, tr2, tr3))
                    else:
                        true_ranges.append(data[j]['high'] - data[j]['low'])
                
                atr_14 = sum(true_ranges) / len(true_ranges)
            else:
                atr_14 = data[i]['high'] - data[i]['low']
            
            # Stochastic Oscillator (14, 3, 3)
            if i >= 13:
                high_14 = max(data[j]['high'] for j in range(i-13, i+1))
                low_14 = min(data[j]['low'] for j in range(i-13, i+1))
                
                if high_14 != low_14:
                    stoch_k = 100 * (data[i]['close'] - low_14) / (high_14 - low_14)
                else:
                    stoch_k = 50
                
                # Simple moving average for %D
                if i >= 16:
                    stoch_d = (indicators[i-2]['stoch_k'] + indicators[i-1]['stoch_k'] + stoch_k) / 3
                else:
                    stoch_d = stoch_k
            else:
                stoch_k = 50
                stoch_d = 50
            
            # Williams %R (14 periods)
            if i >= 13:
                williams_r = -100 * (high_14 - data[i]['close']) / (high_14 - low_14) if high_14 != low_14 else -50
            else:
                williams_r = -50
            
            # CCI (20 periods)
            if i >= 19:
                tp_values = [(data[j]['high'] + data[j]['low'] + data[j]['close']) / 3 
                            for j in range(i-19, i+1)]
                tp_mean = sum(tp_values) / 20
                mean_deviation = sum(abs(tp - tp_mean) for tp in tp_values) / 20
                
                if mean_deviation > 0:
                    cci_20 = (tp_values[-1] - tp_mean) / (0.015 * mean_deviation)
                else:
                    cci_20 = 0
            else:
                cci_20 = 0
            
            indicator_record = {
                'sma_20': round(sma_20, 5),
                'ema_20': round(ema_20, 5),
                'rsi_14': round(rsi_14, 2),
                'macd_line': round(macd_line, 6),
                'bb_upper': round(bb_upper, 5),
                'bb_lower': round(bb_lower, 5),
                'atr_14': round(atr_14, 6),
                'stoch_k': round(stoch_k, 2),
                'stoch_d': round(stoch_d, 2),
                'williams_r': round(williams_r, 2),
                'cci_20': round(cci_20, 2)
            }
            
            indicators.append(indicator_record)
        
        logger.info(f"Calculated technical indicators for {len(indicators)} records")
        return indicators
    
    def save_to_database(self, price_data: List[Dict], indicators: List[Dict]):
        """Save data and indicators to database."""
        logger.info("Saving data to database...")
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Clear existing data
        cursor.execute("DELETE FROM technical_indicators")
        cursor.execute("DELETE FROM price_data")
        
        # Insert price data
        for record in price_data:
            cursor.execute('''
                INSERT INTO price_data (symbol, timestamp, open, high, low, close, volume)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', ('EURUSD=X', record['timestamp'], record['open'], record['high'], 
                  record['low'], record['close'], record['volume']))
        
        conn.commit()
        
        # Get price data IDs and insert indicators
        cursor.execute("SELECT id FROM price_data ORDER BY id")
        price_ids = [row[0] for row in cursor.fetchall()]
        
        for i, indicator in enumerate(indicators):
            if i < len(price_ids):
                cursor.execute('''
                    INSERT INTO technical_indicators 
                    (price_data_id, sma_20, ema_20, rsi_14, macd_line, bb_upper, bb_lower, 
                     atr_14, stoch_k, stoch_d, williams_r, cci_20)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (price_ids[i], indicator['sma_20'], indicator['ema_20'], indicator['rsi_14'],
                      indicator['macd_line'], indicator['bb_upper'], indicator['bb_lower'],
                      indicator['atr_14'], indicator['stoch_k'], indicator['stoch_d'],
                      indicator['williams_r'], indicator['cci_20']))
        
        conn.commit()
        conn.close()
        
        logger.info(f"Saved {len(price_data)} price records and {len(indicators)} indicator records")
    
    def generate_complete_dataset(self, n_records: int = 5000):
        """Generate complete dataset for neural network training."""
        logger.info(f"Generating complete neural network dataset with {n_records} records...")
        
        # Create database
        self.create_database_schema()
        
        # Generate price data
        price_data = self.generate_neural_training_data(n_records)
        
        # Calculate indicators
        indicators = self.calculate_technical_indicators(price_data)
        
        # Save to database
        self.save_to_database(price_data, indicators)
        
        # Create summary
        summary = {
            'timestamp': datetime.now().isoformat(),
            'dataset_type': 'Neural Network Training Data',
            'n_records': len(price_data),
            'symbol': 'EURUSD=X',
            'date_range': {
                'start': price_data[0]['timestamp'],
                'end': price_data[-1]['timestamp']
            },
            'price_statistics': {
                'min_close': min(r['close'] for r in price_data),
                'max_close': max(r['close'] for r in price_data),
                'avg_close': sum(r['close'] for r in price_data) / len(price_data)
            },
            'database_path': self.db_path,
            'tables_created': ['price_data', 'technical_indicators']
        }
        
        # Save summary
        summary_path = 'data/neural_dataset_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Neural network dataset generation completed!")
        logger.info(f"Database: {self.db_path}")
        logger.info(f"Summary: {summary_path}")
        logger.info(f"Records: {n_records}")
        
        return summary


def main():
    """Generate neural network training dataset."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger.info("Starting Neural Network Data Generation...")
    
    # Generate dataset
    generator = SimpleNeuralDataGenerator()
    summary = generator.generate_complete_dataset(5000)
    
    print("\n" + "="*60)
    print("NEURAL NETWORK DATASET GENERATION COMPLETED")
    print("="*60)
    print(f"Records Generated: {summary['n_records']}")
    print(f"Database: {summary['database_path']}")
    print(f"Date Range: {summary['date_range']['start']} to {summary['date_range']['end']}")
    print(f"Price Range: {summary['price_statistics']['min_close']:.5f} - {summary['price_statistics']['max_close']:.5f}")
    print("Tables: price_data, technical_indicators")
    print("\n✅ Ready for Neural Network Training!")
    print("="*60)


if __name__ == "__main__":
    main()