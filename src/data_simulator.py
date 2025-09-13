"""Simulated data generator for development and testing when network access is limited."""

import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from typing import Dict, List, Tuple
import json
from pathlib import Path


class ForexDataSimulator:
    """Generates realistic forex data for development and testing."""
    
    def __init__(self, seed: int = 42):
        """Initialize the simulator with a random seed for reproducibility."""
        np.random.seed(seed)
        self.seed = seed
        
        # Realistic forex pair characteristics
        self.pair_characteristics = {
            "EURUSD=X": {
                "base_price": 1.1000,
                "daily_volatility": 0.005,  # 0.5% daily volatility
                "trend_strength": 0.0001,
                "mean_reversion": 0.95,
                "spread": 0.00015
            },
            "GBPUSD=X": {
                "base_price": 1.2800,
                "daily_volatility": 0.008,  # 0.8% daily volatility
                "trend_strength": 0.00015,
                "mean_reversion": 0.94,
                "spread": 0.0002
            },
            "USDJPY=X": {
                "base_price": 110.00,
                "daily_volatility": 0.006,  # 0.6% daily volatility
                "trend_strength": 0.01,
                "mean_reversion": 0.96,
                "spread": 0.015
            },
            "AUDUSD=X": {
                "base_price": 0.7200,
                "daily_volatility": 0.007,  # 0.7% daily volatility
                "trend_strength": 0.0001,
                "mean_reversion": 0.93,
                "spread": 0.00018
            }
        }
    
    def generate_historical_data(
        self, 
        symbol: str, 
        start_date: datetime, 
        end_date: datetime,
        timeframe: str = "1m"
    ) -> pd.DataFrame:
        """Generate realistic historical forex data."""
        
        if symbol not in self.pair_characteristics:
            raise ValueError(f"Unsupported currency pair: {symbol}")
        
        char = self.pair_characteristics[symbol]
        
        # Calculate time intervals
        if timeframe == "1m":
            freq = "1min"
            periods_per_day = 1440
        elif timeframe == "5m":
            freq = "5min"
            periods_per_day = 288
        elif timeframe == "15m":
            freq = "15min"
            periods_per_day = 96
        elif timeframe == "1h":
            freq = "1H"
            periods_per_day = 24
        elif timeframe == "4h":
            freq = "4H"
            periods_per_day = 6
        elif timeframe == "1d":
            freq = "1D"
            periods_per_day = 1
        else:
            raise ValueError(f"Unsupported timeframe: {timeframe}")
        
        # Generate timestamps
        timestamps = pd.date_range(
            start=start_date,
            end=end_date,
            freq=freq,
            tz=timezone.utc
        )
        
        # Filter out weekends for forex (market closed)
        weekday_timestamps = timestamps[timestamps.weekday < 5]
        
        n_periods = len(weekday_timestamps)
        if n_periods == 0:
            return pd.DataFrame()
        
        # Generate price series using geometric brownian motion with mean reversion
        prices = []
        current_price = char["base_price"]
        long_term_mean = current_price
        
        # Scale volatility to timeframe
        if timeframe == "1m":
            vol_scale = 1.0
        elif timeframe == "5m":
            vol_scale = np.sqrt(5)
        elif timeframe == "15m":
            vol_scale = np.sqrt(15)
        elif timeframe == "1h":
            vol_scale = np.sqrt(60)
        elif timeframe == "4h":
            vol_scale = np.sqrt(240)
        elif timeframe == "1d":
            vol_scale = np.sqrt(1440)
        
        volatility = char["daily_volatility"] / np.sqrt(periods_per_day) * vol_scale
        
        for i in range(n_periods):
            # Mean reversion component
            mean_revert = char["mean_reversion"] * (long_term_mean - current_price) / current_price
            
            # Random walk component
            random_shock = np.random.normal(0, volatility)
            
            # Trend component (small random walk for long-term trend)
            if i % (periods_per_day * 7) == 0:  # Weekly trend adjustment
                trend_adj = np.random.normal(0, char["trend_strength"])
                long_term_mean *= (1 + trend_adj)
            
            # Update price
            price_change = mean_revert + random_shock
            current_price *= (1 + price_change)
            
            # Ensure price stays within reasonable bounds
            if current_price < char["base_price"] * 0.7:
                current_price = char["base_price"] * 0.7
            elif current_price > char["base_price"] * 1.3:
                current_price = char["base_price"] * 1.3
                
            prices.append(current_price)
        
        # Generate OHLC data from price series
        ohlc_data = []
        for i, (timestamp, close_price) in enumerate(zip(weekday_timestamps, prices)):
            
            # Generate intraperiod volatility for OHLC
            if i == 0:
                open_price = close_price
            else:
                open_price = prices[i-1]
            
            # High/Low generation
            period_vol = volatility * np.random.uniform(0.3, 1.5)
            high_factor = 1 + abs(np.random.normal(0, period_vol))
            low_factor = 1 - abs(np.random.normal(0, period_vol))
            
            high_price = max(open_price, close_price) * high_factor
            low_price = min(open_price, close_price) * low_factor
            
            # Ensure OHLC consistency
            high_price = max(high_price, open_price, close_price)
            low_price = min(low_price, open_price, close_price)
            
            # Generate realistic volume (forex doesn't have true volume, so use tick volume)
            base_volume = np.random.lognormal(8, 1.5)  # Log-normal distribution
            if timestamp.hour in [8, 9, 13, 14, 15, 16]:  # Active trading hours
                base_volume *= np.random.uniform(1.5, 3.0)
            
            ohlc_data.append({
                "timestamp": timestamp,
                "Open": open_price,
                "High": high_price,
                "Low": low_price,
                "Close": close_price,
                "Volume": int(base_volume)
            })
        
        # Create DataFrame
        df = pd.DataFrame(ohlc_data)
        df.set_index("timestamp", inplace=True)
        
        return df
    
    def generate_sample_datasets(self, data_dir: Path) -> Dict[str, Dict]:
        """Generate sample datasets for all currency pairs."""
        
        data_dir = Path(data_dir)
        data_dir.mkdir(exist_ok=True)
        
        results = {}
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(days=365 * 2)  # 2 years of data
        
        for symbol in self.pair_characteristics.keys():
            try:
                # Generate 1-minute data
                data_1m = self.generate_historical_data(symbol, start_date, end_date, "1m")
                
                # Save to CSV for inspection
                csv_file = data_dir / f"{symbol.replace('=', '_')}_1m_simulated.csv"
                data_1m.to_csv(csv_file)
                
                # Calculate quality metrics
                quality_metrics = self._calculate_quality_metrics(data_1m, symbol)
                
                results[symbol] = {
                    "status": "success",
                    "records_generated": len(data_1m),
                    "start_date": data_1m.index[0].isoformat() if len(data_1m) > 0 else None,
                    "end_date": data_1m.index[-1].isoformat() if len(data_1m) > 0 else None,
                    "quality_metrics": quality_metrics,
                    "file_path": str(csv_file)
                }
                
            except Exception as e:
                results[symbol] = {
                    "status": "error",
                    "error": str(e)
                }
        
        return results
    
    def _calculate_quality_metrics(self, data: pd.DataFrame, symbol: str) -> Dict:
        """Calculate quality metrics for generated data."""
        if len(data) == 0:
            return {"quality_score": 0.0}
        
        total_points = len(data)
        
        # Check for missing values (should be none in simulated data)
        missing_values = data.isnull().sum().sum()
        completeness = 1 - (missing_values / (total_points * len(data.columns)))
        
        # Check for realistic price movements (no gaps should exist)
        gap_rate = 0.0  # Simulated data has no gaps
        
        # Check for price consistency (OHLC relationships)
        invalid_ohlc = 0
        for _, row in data.iterrows():
            if not (row['Low'] <= row['Open'] <= row['High'] and
                   row['Low'] <= row['Close'] <= row['High']):
                invalid_ohlc += 1
        
        ohlc_consistency = 1 - (invalid_ohlc / total_points)
        
        # Overall quality score
        quality_score = (completeness * 0.4 + (1 - gap_rate) * 0.3 + ohlc_consistency * 0.3) * 100
        
        return {
            "quality_score": round(quality_score, 2),
            "completeness": round(completeness * 100, 2),
            "gap_rate": round(gap_rate * 100, 2),
            "ohlc_consistency": round(ohlc_consistency * 100, 2),
            "total_records": total_points,
            "data_type": "simulated"
        }


def main():
    """Generate sample data for development."""
    print("🔄 Generating simulated forex data for development...")
    
    simulator = ForexDataSimulator(seed=42)
    data_dir = Path("data/simulated")
    
    results = simulator.generate_sample_datasets(data_dir)
    
    print("\n=== Simulated Data Generation Results ===")
    total_records = 0
    
    for symbol, result in results.items():
        print(f"\n{symbol}:")
        print(f"  Status: {result['status']}")
        
        if result['status'] == 'success':
            print(f"  Records Generated: {result['records_generated']:,}")
            print(f"  Quality Score: {result['quality_metrics']['quality_score']}%")
            print(f"  Date Range: {result['start_date']} to {result['end_date']}")
            print(f"  File: {result['file_path']}")
            total_records += result['records_generated']
        else:
            print(f"  Error: {result['error']}")
    
    print(f"\n📊 Total Records Generated: {total_records:,}")
    print(f"💾 Data Directory: {data_dir.absolute()}")
    print("\n✅ Simulated data generation completed!")
    
    return results


if __name__ == "__main__":
    main()