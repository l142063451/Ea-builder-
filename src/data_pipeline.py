"""Data collection pipeline for forex data using yfinance."""

import asyncio
import time
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from typing import List, Dict, Optional, Tuple
import pandas as pd
import yfinance as yf
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import IntegrityError

from .config import settings
from .database import Base, Currency, PriceData
from .logger import trading_logger

class DataCollector:
    """Main data collection class for forex data."""
    
    def __init__(self):
        self.engine = create_engine(settings.database.url, echo=False)
        Base.metadata.create_all(bind=self.engine)
        SessionLocal = sessionmaker(bind=self.engine)
        self.db = SessionLocal()
        
        # Initialize currency pairs
        self._initialize_currencies()
        
        trading_logger.log_system_event(
            event_type="data_collector_init",
            component="DataCollector",
            status="initialized",
            details={"database_url": settings.database.url}
        )
    
    def _initialize_currencies(self) -> None:
        """Initialize currency pairs in the database."""
        currency_configs = [
            {"symbol": "EURUSD=X", "name": "Euro/US Dollar", "base": "EUR", "quote": "USD"},
            {"symbol": "GBPUSD=X", "name": "British Pound/US Dollar", "base": "GBP", "quote": "USD"},
            {"symbol": "USDJPY=X", "name": "US Dollar/Japanese Yen", "base": "USD", "quote": "JPY", "tick_size": 0.001},
            {"symbol": "AUDUSD=X", "name": "Australian Dollar/US Dollar", "base": "AUD", "quote": "USD"}
        ]
        
        for config in currency_configs:
            existing = self.db.query(Currency).filter(Currency.symbol == config["symbol"]).first()
            if not existing:
                currency = Currency(
                    symbol=config["symbol"],
                    name=config["name"],
                    base_currency=config["base"],
                    quote_currency=config["quote"],
                    tick_size=Decimal(str(config.get("tick_size", 0.00001)))
                )
                self.db.add(currency)
        
        self.db.commit()
        trading_logger.log_system_event(
            event_type="currencies_initialized",
            component="DataCollector",
            status="completed"
        )
    
    def download_historical_data(
        self, 
        symbol: str, 
        period: str = "5y", 
        interval: str = "1m"
    ) -> pd.DataFrame:
        """Download historical data for a currency pair."""
        try:
            trading_logger.logger.info(
                f"Downloading historical data for {symbol}",
                symbol=symbol,
                period=period,
                interval=interval
            )
            
            ticker = yf.Ticker(symbol)
            data = ticker.history(
                period=period,
                interval=interval,
                actions=False,
                auto_adjust=False,
                back_adjust=False
            )
            
            if data.empty:
                raise ValueError(f"No data received for {symbol}")
            
            # Clean the data
            data = data.dropna()
            data.index = pd.to_datetime(data.index, utc=True)
            
            trading_logger.logger.info(
                f"Successfully downloaded {len(data)} records for {symbol}",
                symbol=symbol,
                records_count=len(data),
                start_date=str(data.index[0]),
                end_date=str(data.index[-1])
            )
            
            return data
            
        except Exception as e:
            trading_logger.logger.error(
                f"Failed to download data for {symbol}",
                symbol=symbol,
                error=str(e)
            )
            raise
    
    def validate_data_quality(self, data: pd.DataFrame, symbol: str) -> Dict[str, float]:
        """Validate data quality and return quality metrics."""
        total_points = len(data)
        
        if total_points == 0:
            return {"quality_score": 0.0, "completeness": 0.0, "gaps": 1.0}
        
        # Check for missing values
        missing_values = data.isnull().sum().sum()
        completeness = 1 - (missing_values / (total_points * len(data.columns)))
        
        # Check for price gaps (weekends excluded)
        data_sorted = data.sort_index()
        time_diffs = data_sorted.index.to_series().diff()
        expected_interval = pd.Timedelta(minutes=1)  # Assuming 1-minute data
        
        # Count gaps > 1 hour (excluding weekends)
        large_gaps = 0
        weekend_gaps = 0
        
        for i, diff in enumerate(time_diffs):
            if pd.isna(diff):
                continue
            
            if diff > pd.Timedelta(hours=1):
                # Check if it's a weekend gap
                prev_time = data_sorted.index[i-1]
                curr_time = data_sorted.index[i]
                
                if prev_time.weekday() >= 5 or curr_time.weekday() >= 5:
                    weekend_gaps += 1
                else:
                    large_gaps += 1
        
        gap_rate = large_gaps / max(total_points, 1)
        
        # Check for outliers (using IQR method)
        outliers = 0
        for col in ['Open', 'High', 'Low', 'Close']:
            if col in data.columns:
                Q1 = data[col].quantile(0.25)
                Q3 = data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outliers += len(data[(data[col] < lower_bound) | (data[col] > upper_bound)])
        
        outlier_rate = outliers / (total_points * 4)  # 4 price columns
        
        # Calculate overall quality score
        quality_score = (
            completeness * 0.4 +
            (1 - gap_rate) * 0.3 +
            (1 - outlier_rate) * 0.3
        )
        
        quality_metrics = {
            "quality_score": round(quality_score * 100, 2),
            "completeness": round(completeness * 100, 2),
            "gap_rate": round(gap_rate * 100, 2),
            "outlier_rate": round(outlier_rate * 100, 2),
            "total_records": total_points,
            "weekend_gaps": weekend_gaps,
            "data_gaps": large_gaps
        }
        
        trading_logger.log_data_quality(
            source=f"yfinance_{symbol}",
            quality_score=quality_score * 100,
            issues=quality_metrics
        )
        
        return quality_metrics
    
    def store_price_data(
        self, 
        data: pd.DataFrame, 
        symbol: str, 
        timeframe: str = "1m"
    ) -> int:
        """Store price data in the database."""
        currency = self.db.query(Currency).filter(Currency.symbol == symbol).first()
        if not currency:
            raise ValueError(f"Currency {symbol} not found in database")
        
        records_inserted = 0
        batch_size = 1000
        
        try:
            for i in range(0, len(data), batch_size):
                batch = data.iloc[i:i + batch_size]
                price_records = []
                
                for timestamp, row in batch.iterrows():
                    price_record = PriceData(
                        currency_id=currency.id,
                        timestamp=timestamp,
                        timeframe=timeframe,
                        open_price=Decimal(str(row['Open'])),
                        high_price=Decimal(str(row['High'])),
                        low_price=Decimal(str(row['Low'])),
                        close_price=Decimal(str(row['Close'])),
                        volume=Decimal(str(row['Volume']) if 'Volume' in row else '0')
                    )
                    price_records.append(price_record)
                
                # Bulk insert with duplicate handling
                try:
                    self.db.bulk_save_objects(price_records)
                    self.db.commit()
                    records_inserted += len(price_records)
                    
                except IntegrityError:
                    # Handle duplicates by inserting one by one
                    self.db.rollback()
                    for record in price_records:
                        try:
                            self.db.add(record)
                            self.db.commit()
                            records_inserted += 1
                        except IntegrityError:
                            self.db.rollback()
                            continue
                
                # Log progress
                if i % (batch_size * 10) == 0:
                    trading_logger.logger.info(
                        f"Stored {records_inserted} records for {symbol}",
                        symbol=symbol,
                        progress=f"{i + batch_size}/{len(data)}"
                    )
        
        except Exception as e:
            self.db.rollback()
            trading_logger.logger.error(
                f"Failed to store data for {symbol}",
                symbol=symbol,
                error=str(e)
            )
            raise
        
        trading_logger.logger.info(
            f"Successfully stored {records_inserted} records for {symbol}",
            symbol=symbol,
            total_records=records_inserted
        )
        
        return records_inserted
    
    def collect_all_currencies(self) -> Dict[str, Dict]:
        """Collect historical data for all configured currency pairs."""
        results = {}
        
        for symbol in settings.data.currency_pairs:
            try:
                start_time = time.time()
                
                # Download data
                data = self.download_historical_data(
                    symbol=symbol,
                    period=f"{settings.data.historical_data_years}y",
                    interval="1m"
                )
                
                # Validate data quality
                quality_metrics = self.validate_data_quality(data, symbol)
                
                # Store in database if quality is acceptable
                if quality_metrics["quality_score"] >= 95.0:  # 95% threshold
                    records_stored = self.store_price_data(data, symbol, "1m")
                    
                    end_time = time.time()
                    processing_time = end_time - start_time
                    
                    results[symbol] = {
                        "status": "success",
                        "records_downloaded": len(data),
                        "records_stored": records_stored,
                        "quality_metrics": quality_metrics,
                        "processing_time_seconds": round(processing_time, 2)
                    }
                    
                else:
                    results[symbol] = {
                        "status": "quality_failed",
                        "quality_metrics": quality_metrics,
                        "reason": f"Quality score {quality_metrics['quality_score']}% below 95% threshold"
                    }
                    
            except Exception as e:
                results[symbol] = {
                    "status": "error",
                    "error": str(e)
                }
                
                trading_logger.logger.error(
                    f"Failed to collect data for {symbol}",
                    symbol=symbol,
                    error=str(e)
                )
        
        return results
    
    def get_data_summary(self) -> Dict[str, Dict]:
        """Get summary of stored data for all currency pairs."""
        summary = {}
        
        currencies = self.db.query(Currency).filter(Currency.is_active == True).all()
        
        for currency in currencies:
            count = self.db.query(PriceData).filter(
                PriceData.currency_id == currency.id,
                PriceData.timeframe == "1m"
            ).count()
            
            if count > 0:
                first_record = self.db.query(PriceData).filter(
                    PriceData.currency_id == currency.id,
                    PriceData.timeframe == "1m"
                ).order_by(PriceData.timestamp.asc()).first()
                
                last_record = self.db.query(PriceData).filter(
                    PriceData.currency_id == currency.id,
                    PriceData.timeframe == "1m"
                ).order_by(PriceData.timestamp.desc()).first()
                
                summary[currency.symbol] = {
                    "total_records": count,
                    "first_timestamp": first_record.timestamp.isoformat() if first_record else None,
                    "last_timestamp": last_record.timestamp.isoformat() if last_record else None,
                    "data_span_days": (last_record.timestamp - first_record.timestamp).days if first_record and last_record else 0
                }
            else:
                summary[currency.symbol] = {
                    "total_records": 0,
                    "first_timestamp": None,
                    "last_timestamp": None,
                    "data_span_days": 0
                }
        
        return summary
    
    def close(self) -> None:
        """Close database connection."""
        self.db.close()

def main():
    """Main function to run data collection."""
    collector = DataCollector()
    
    try:
        trading_logger.logger.info("Starting historical data collection")
        
        # Collect all currency data
        results = collector.collect_all_currencies()
        
        # Print summary
        print("\n=== Data Collection Results ===")
        for symbol, result in results.items():
            print(f"\n{symbol}:")
            print(f"  Status: {result['status']}")
            
            if result['status'] == 'success':
                print(f"  Records Downloaded: {result['records_downloaded']:,}")
                print(f"  Records Stored: {result['records_stored']:,}")
                print(f"  Quality Score: {result['quality_metrics']['quality_score']}%")
                print(f"  Processing Time: {result['processing_time_seconds']}s")
            elif result['status'] == 'quality_failed':
                print(f"  Quality Score: {result['quality_metrics']['quality_score']}%")
                print(f"  Reason: {result['reason']}")
            else:
                print(f"  Error: {result['error']}")
        
        # Get data summary
        summary = collector.get_data_summary()
        print("\n=== Stored Data Summary ===")
        for symbol, data in summary.items():
            print(f"\n{symbol}:")
            print(f"  Total Records: {data['total_records']:,}")
            print(f"  Date Range: {data['first_timestamp']} to {data['last_timestamp']}")
            print(f"  Data Span: {data['data_span_days']} days")
        
        trading_logger.logger.info("Data collection completed", results=results)
        
    except Exception as e:
        trading_logger.logger.error("Data collection failed", error=str(e))
        raise
    finally:
        collector.close()

if __name__ == "__main__":
    main()