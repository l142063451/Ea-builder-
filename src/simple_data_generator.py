"""
Simple Data Generator for Baseline Models

Creates realistic forex data with technical indicators using only standard library.
This generates better quality training data for the baseline models.
"""

import sqlite3
import math
import random
import json
import os
from typing import List, Tuple, Dict
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class SimpleForexGenerator:
    """
    Generates realistic forex price data with technical indicators.
    Uses geometric Brownian motion with market microstructure.
    """
    
    def __init__(self, symbol: str = "EURUSD=X", base_price: float = 1.1000):
        self.symbol = symbol
        self.base_price = base_price
        self.current_price = base_price
        
    def generate_price_data(self, n_samples: int = 5000) -> List[Dict]:
        """Generate realistic OHLCV price data."""
        data = []
        current_time = datetime(2024, 1, 1)
        
        # Market parameters
        volatility = 0.0001  # Daily volatility ~0.01%
        trend = 0.000001     # Slight upward trend
        mean_reversion = 0.001  # Mean reversion strength
        
        for i in range(n_samples):
            # Generate random walk with mean reversion
            random_change = (random.random() - 0.5) * volatility
            
            # Add trend component
            trend_component = trend * (1 + 0.1 * math.sin(i / 100))
            
            # Mean reversion to base price
            reversion = -mean_reversion * (self.current_price - self.base_price)
            
            # Market microstructure (bid-ask spread effects)
            microstructure = (random.random() - 0.5) * 0.00005
            
            # Total price change
            price_change = random_change + trend_component + reversion + microstructure
            
            # Update current price
            self.current_price += price_change
            
            # Generate OHLCV
            open_price = self.current_price
            high_change = abs(random.random()) * volatility * 2
            low_change = -abs(random.random()) * volatility * 2
            
            high = open_price + high_change
            low = open_price + low_change
            close = open_price + (random.random() - 0.5) * volatility
            volume = random.randint(1000, 10000)
            
            # Ensure OHLC consistency
            high = max(high, open_price, close)
            low = min(low, open_price, close)
            
            record = {
                'timestamp': current_time.strftime('%Y-%m-%d %H:%M:%S'),
                'open': round(open_price, 5),
                'high': round(high, 5),
                'low': round(low, 5),
                'close': round(close, 5),
                'volume': volume
            }
            
            data.append(record)
            current_time += timedelta(minutes=1)
        
        return data
    
    def calculate_technical_indicators(self, data: List[Dict]) -> List[Dict]:
        """Calculate technical indicators for the price data."""
        logger.info("Calculating technical indicators...")
        
        closes = [d['close'] for d in data]
        highs = [d['high'] for d in data]
        lows = [d['low'] for d in data]
        volumes = [d['volume'] for d in data]
        
        enhanced_data = []
        
        for i, record in enumerate(data):
            enhanced_record = record.copy()
            
            # Simple Moving Average (20 period)
            if i >= 19:
                sma_20 = sum(closes[i-19:i+1]) / 20
                enhanced_record['sma_20'] = round(sma_20, 5)
            else:
                enhanced_record['sma_20'] = closes[i]
            
            # Exponential Moving Average (20 period)
            if i == 0:
                enhanced_record['ema_20'] = closes[i]
            else:
                alpha = 2 / (20 + 1)
                prev_ema = enhanced_data[i-1]['ema_20'] if i > 0 else closes[i]
                ema_20 = alpha * closes[i] + (1 - alpha) * prev_ema
                enhanced_record['ema_20'] = round(ema_20, 5)
            
            # RSI (14 period)
            if i >= 14:
                gains = []
                losses = []
                for j in range(i-13, i+1):
                    change = closes[j] - closes[j-1] if j > 0 else 0
                    if change > 0:
                        gains.append(change)
                        losses.append(0)
                    else:
                        gains.append(0)
                        losses.append(abs(change))
                
                avg_gain = sum(gains) / len(gains)
                avg_loss = sum(losses) / len(losses)
                
                if avg_loss == 0:
                    rsi = 100
                else:
                    rs = avg_gain / avg_loss
                    rsi = 100 - (100 / (1 + rs))
                
                enhanced_record['rsi_14'] = round(rsi, 2)
            else:
                enhanced_record['rsi_14'] = 50.0
            
            # MACD
            if i >= 25:  # Need 26 periods for MACD
                # 12-period EMA
                ema_12 = closes[i]
                for j in range(1, min(12, i+1)):
                    alpha = 2 / (12 + 1)
                    ema_12 = alpha * closes[i-j] + (1 - alpha) * ema_12
                
                # 26-period EMA
                ema_26 = closes[i]
                for j in range(1, min(26, i+1)):
                    alpha = 2 / (26 + 1)
                    ema_26 = alpha * closes[i-j] + (1 - alpha) * ema_26
                
                macd_line = ema_12 - ema_26
                enhanced_record['macd_line'] = round(macd_line, 6)
            else:
                enhanced_record['macd_line'] = 0.0
            
            # Bollinger Bands (20 period, 2 std dev)
            if i >= 19:
                sma_20 = sum(closes[i-19:i+1]) / 20
                variance = sum((close - sma_20) ** 2 for close in closes[i-19:i+1]) / 20
                std_dev = math.sqrt(variance)
                
                bb_upper = sma_20 + (2 * std_dev)
                bb_lower = sma_20 - (2 * std_dev)
                
                enhanced_record['bb_upper'] = round(bb_upper, 5)
                enhanced_record['bb_lower'] = round(bb_lower, 5)
            else:
                enhanced_record['bb_upper'] = closes[i] * 1.01
                enhanced_record['bb_lower'] = closes[i] * 0.99
            
            # Average True Range (14 period)
            if i >= 14:
                true_ranges = []
                for j in range(i-13, i+1):
                    if j > 0:
                        tr1 = highs[j] - lows[j]
                        tr2 = abs(highs[j] - closes[j-1])
                        tr3 = abs(lows[j] - closes[j-1])
                        true_range = max(tr1, tr2, tr3)
                    else:
                        true_range = highs[j] - lows[j]
                    true_ranges.append(true_range)
                
                atr = sum(true_ranges) / len(true_ranges)
                enhanced_record['atr_14'] = round(atr, 6)
            else:
                enhanced_record['atr_14'] = abs(highs[i] - lows[i])
            
            # Stochastic Oscillator (14 period)
            if i >= 13:
                lowest_low = min(lows[i-13:i+1])
                highest_high = max(highs[i-13:i+1])
                
                if highest_high - lowest_low == 0:
                    stoch_k = 50
                else:
                    stoch_k = ((closes[i] - lowest_low) / (highest_high - lowest_low)) * 100
                
                enhanced_record['stoch_k'] = round(stoch_k, 2)
                
                # Stoch %D (3-period SMA of %K)
                if i >= 15:
                    recent_k_values = []
                    for j in range(max(0, i-2), i+1):
                        if j >= 13:
                            ll = min(lows[j-13:j+1])
                            hh = max(highs[j-13:j+1])
                            if hh - ll != 0:
                                k_val = ((closes[j] - ll) / (hh - ll)) * 100
                            else:
                                k_val = 50
                            recent_k_values.append(k_val)
                    
                    stoch_d = sum(recent_k_values) / len(recent_k_values)
                    enhanced_record['stoch_d'] = round(stoch_d, 2)
                else:
                    enhanced_record['stoch_d'] = stoch_k
            else:
                enhanced_record['stoch_k'] = 50.0
                enhanced_record['stoch_d'] = 50.0
            
            # Williams %R (14 period)
            if i >= 13:
                lowest_low = min(lows[i-13:i+1])
                highest_high = max(highs[i-13:i+1])
                
                if highest_high - lowest_low == 0:
                    williams_r = -50
                else:
                    williams_r = ((highest_high - closes[i]) / (highest_high - lowest_low)) * -100
                
                enhanced_record['williams_r'] = round(williams_r, 2)
            else:
                enhanced_record['williams_r'] = -50.0
            
            # Commodity Channel Index (20 period)
            if i >= 19:
                typical_prices = [(highs[j] + lows[j] + closes[j]) / 3 for j in range(i-19, i+1)]
                sma_tp = sum(typical_prices) / 20
                
                mean_deviation = sum(abs(tp - sma_tp) for tp in typical_prices) / 20
                
                if mean_deviation == 0:
                    cci = 0
                else:
                    current_tp = (highs[i] + lows[i] + closes[i]) / 3
                    cci = (current_tp - sma_tp) / (0.015 * mean_deviation)
                
                enhanced_record['cci_20'] = round(cci, 2)
            else:
                enhanced_record['cci_20'] = 0.0
            
            enhanced_data.append(enhanced_record)
        
        return enhanced_data


class SimpleDatabaseManager:
    """Manages SQLite database operations without external dependencies."""
    
    def __init__(self, db_path: str = "data/trading_bot.db"):
        self.db_path = db_path
        self.ensure_directory()
    
    def ensure_directory(self):
        """Ensure the data directory exists."""
        os.makedirs(os.path.dirname(self.db_path) if os.path.dirname(self.db_path) else '.', exist_ok=True)
    
    def create_tables(self):
        """Create database tables."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create price_data table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS price_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            timestamp DATETIME NOT NULL,
            open REAL NOT NULL,
            high REAL NOT NULL,
            low REAL NOT NULL,
            close REAL NOT NULL,
            volume INTEGER NOT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(symbol, timestamp)
        )
        ''')
        
        # Create technical_indicators table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS technical_indicators (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            price_data_id INTEGER NOT NULL,
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
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (price_data_id) REFERENCES price_data(id),
            UNIQUE(price_data_id)
        )
        ''')
        
        conn.commit()
        conn.close()
        logger.info("Database tables created successfully")
    
    def insert_data(self, symbol: str, data: List[Dict]):
        """Insert price data and technical indicators."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        inserted_count = 0
        
        for record in data:
            try:
                # Insert price data
                cursor.execute('''
                INSERT OR REPLACE INTO price_data 
                (symbol, timestamp, open, high, low, close, volume)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    symbol,
                    record['timestamp'],
                    record['open'],
                    record['high'],
                    record['low'],
                    record['close'],
                    record['volume']
                ))
                
                price_data_id = cursor.lastrowid
                
                # Insert technical indicators
                cursor.execute('''
                INSERT OR REPLACE INTO technical_indicators
                (price_data_id, sma_20, ema_20, rsi_14, macd_line, bb_upper, bb_lower,
                 atr_14, stoch_k, stoch_d, williams_r, cci_20)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    price_data_id,
                    record.get('sma_20'),
                    record.get('ema_20'),
                    record.get('rsi_14'),
                    record.get('macd_line'),
                    record.get('bb_upper'),
                    record.get('bb_lower'),
                    record.get('atr_14'),
                    record.get('stoch_k'),
                    record.get('stoch_d'),
                    record.get('williams_r'),
                    record.get('cci_20')
                ))
                
                inserted_count += 1
                
            except sqlite3.IntegrityError:
                # Record already exists
                pass
        
        conn.commit()
        conn.close()
        
        logger.info(f"Inserted {inserted_count} records for {symbol}")
        return inserted_count


def generate_forex_dataset():
    """Generate complete forex dataset with multiple currency pairs."""
    logger.info("Starting forex dataset generation...")
    
    # Initialize database
    db_manager = SimpleDatabaseManager()
    db_manager.create_tables()
    
    # Currency pairs to generate
    pairs = [
        ("EURUSD=X", 1.1000),
        ("GBPUSD=X", 1.2700),
        ("USDJPY=X", 148.50),
        ("AUDUSD=X", 0.6700)
    ]
    
    total_records = 0
    results = {}
    
    for symbol, base_price in pairs:
        logger.info(f"Generating data for {symbol}...")
        
        # Generate data
        generator = SimpleForexGenerator(symbol, base_price)
        price_data = generator.generate_price_data(n_samples=2500)  # ~1.7 days of 1-minute data
        enhanced_data = generator.calculate_technical_indicators(price_data)
        
        # Insert into database
        inserted = db_manager.insert_data(symbol, enhanced_data)
        total_records += inserted
        
        # Store results
        results[symbol] = {
            'status': 'success',
            'records_generated': len(enhanced_data),
            'records_stored': inserted,
            'base_price': base_price,
            'start_date': enhanced_data[0]['timestamp'],
            'end_date': enhanced_data[-1]['timestamp']
        }
        
        logger.info(f"Completed {symbol}: {inserted} records")
    
    # Save generation summary
    summary = {
        'generation_timestamp': datetime.now().isoformat(),
        'total_records': total_records,
        'pairs_generated': len(pairs),
        'database_path': db_manager.db_path,
        'results': results
    }
    
    summary_path = "data/generation_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"Dataset generation complete: {total_records} total records")
    logger.info(f"Summary saved to {summary_path}")
    
    return summary


def main():
    """Main function to generate forex dataset."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger.info("Simple forex data generator starting...")
    
    try:
        summary = generate_forex_dataset()
        
        print("\n" + "="*60)
        print("FOREX DATA GENERATION COMPLETED")
        print("="*60)
        print(f"Total Records Generated: {summary['total_records']}")
        print(f"Database Location: {summary['database_path']}")
        print(f"Currency Pairs: {summary['pairs_generated']}")
        print()
        
        for symbol, result in summary['results'].items():
            print(f"{symbol}:")
            print(f"  Records: {result['records_stored']}")
            print(f"  Period: {result['start_date']} to {result['end_date']}")
        
        print("\n✅ READY FOR BASELINE MODEL TRAINING")
        print("="*60)
        
    except Exception as e:
        logger.error(f"Error generating dataset: {e}")
        raise


if __name__ == "__main__":
    main()