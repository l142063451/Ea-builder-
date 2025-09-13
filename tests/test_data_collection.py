"""Simple test of data collection functionality with SQLite."""

import os
import sys
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Set up SQLite database URL for testing
os.environ['DATABASE_URL'] = 'sqlite:///test_forex.db'

try:
    from src.data_pipeline import DataCollector
    from src.logger import trading_logger
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)

def test_yfinance_connection():
    """Test basic yfinance connectivity."""
    import yfinance as yf
    
    print("Testing yfinance connectivity...")
    try:
        # Test with a simple request
        ticker = yf.Ticker("EURUSD=X")
        # Get just 1 day of data to test connectivity
        data = ticker.history(period="1d", interval="1h")
        
        if not data.empty:
            print(f"✅ yfinance connection successful - got {len(data)} records")
            print(f"Latest price: {data['Close'].iloc[-1]:.5f}")
            return True
        else:
            print("❌ yfinance returned empty data")
            return False
            
    except Exception as e:
        print(f"❌ yfinance connection failed: {e}")
        return False

def test_database_setup():
    """Test database setup with SQLite."""
    print("\nTesting database setup...")
    try:
        collector = DataCollector()
        summary = collector.get_data_summary()
        print(f"✅ Database setup successful - found {len(summary)} currency pairs")
        collector.close()
        return True
    except Exception as e:
        print(f"❌ Database setup failed: {e}")
        return False

def test_small_data_collection():
    """Test collecting a small amount of data."""
    print("\nTesting small data collection...")
    try:
        collector = DataCollector()
        
        # Download just 1 day of EURUSD data
        data = collector.download_historical_data("EURUSD=X", period="1d", interval="5m")
        
        if not data.empty:
            print(f"✅ Downloaded {len(data)} records for EURUSD")
            
            # Test data validation
            quality_metrics = collector.validate_data_quality(data, "EURUSD=X")
            print(f"✅ Data quality score: {quality_metrics['quality_score']}%")
            
            # Test storing data (small amount)
            if len(data) < 500:  # Only store if it's a small dataset
                records_stored = collector.store_price_data(data, "EURUSD=X", "5m")
                print(f"✅ Stored {records_stored} records in database")
            else:
                print(f"⚠️  Skipping storage - dataset too large ({len(data)} records)")
            
            collector.close()
            return True
        else:
            print("❌ No data downloaded")
            collector.close()
            return False
            
    except Exception as e:
        print(f"❌ Data collection test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("=== Forex Trading Bot - Data Collection Test ===")
    print(f"Test started at: {datetime.now(timezone.utc).isoformat()}")
    
    tests = [
        test_yfinance_connection,
        test_database_setup,
        test_small_data_collection
    ]
    
    results = []
    for test_func in tests:
        try:
            result = test_func()
            results.append(result)
        except Exception as e:
            print(f"❌ Test {test_func.__name__} crashed: {e}")
            results.append(False)
    
    print(f"\n=== Test Summary ===")
    passed = sum(results)
    total = len(results)
    print(f"Tests Passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed! Data collection pipeline is working.")
        return True
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)