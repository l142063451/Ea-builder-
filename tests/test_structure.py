"""Basic structure validation test for the Forex Trading Bot."""

import os
import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_imports():
    """Test that all core modules can be imported."""
    print("Testing core module imports...")
    
    try:
        # Test configuration
        from src.config import settings
        print("✅ Config module imported successfully")
        
        # Test database models
        from src.database import Base, Currency, PriceData, TradingSignal, Trade
        print("✅ Database models imported successfully")
        
        # Test logger
        from src.logger import trading_logger, setup_logging
        print("✅ Logger module imported successfully")
        
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False

def test_configuration():
    """Test configuration loading."""
    print("\nTesting configuration...")
    
    try:
        from src.config import settings
        
        # Test database config
        assert hasattr(settings, 'database')
        assert hasattr(settings, 'trading')
        assert hasattr(settings, 'data')
        assert hasattr(settings, 'risk')
        
        print(f"✅ Database URL: {settings.database.url}")
        print(f"✅ Demo mode: {settings.trading.demo_mode}")
        print(f"✅ Currency pairs: {len(settings.data.currency_pairs)}")
        print(f"✅ Target accuracy: {settings.performance.target_accuracy}")
        
        return True
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

def test_database_models():
    """Test database model creation."""
    print("\nTesting database models...")
    
    try:
        # Set up SQLite for testing
        os.environ['DATABASE_URL'] = 'sqlite:///test_structure.db'
        
        from src.database import Base, Currency, PriceData
        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker
        
        # Create in-memory database for testing
        engine = create_engine('sqlite:///:memory:', echo=False)
        Base.metadata.create_all(bind=engine)
        
        SessionLocal = sessionmaker(bind=engine)
        db = SessionLocal()
        
        # Test creating a currency
        currency = Currency(
            symbol="TEST=X",
            name="Test Currency",
            base_currency="TEST",
            quote_currency="USD"
        )
        db.add(currency)
        db.commit()
        
        # Verify it was created
        retrieved = db.query(Currency).filter(Currency.symbol == "TEST=X").first()
        assert retrieved is not None
        assert retrieved.name == "Test Currency"
        
        db.close()
        print("✅ Database models work correctly")
        return True
        
    except Exception as e:
        print(f"❌ Database model test failed: {e}")
        return False

def test_logging_setup():
    """Test logging system."""
    print("\nTesting logging system...")
    
    try:
        from src.logger import trading_logger, setup_logging
        
        # Test logger creation
        logger = setup_logging()
        
        # Test trading logger methods
        trading_logger.log_system_event(
            event_type="test",
            component="structure_test",
            status="testing"
        )
        
        print("✅ Logging system initialized successfully")
        return True
        
    except Exception as e:
        print(f"❌ Logging test failed: {e}")
        return False

def test_project_structure():
    """Test project directory structure."""
    print("\nTesting project structure...")
    
    required_dirs = [
        'src',
        'tests', 
        'data',
        'logs',
        'config',
        'examples'
    ]
    
    required_files = [
        'src/__init__.py',
        'src/config.py',
        'src/database.py',
        'src/logger.py',
        'src/data_pipeline.py',
        'requirements.txt',
        'pyproject.toml'
    ]
    
    missing_dirs = []
    missing_files = []
    
    for dir_name in required_dirs:
        if not (project_root / dir_name).exists():
            missing_dirs.append(dir_name)
    
    for file_name in required_files:
        if not (project_root / file_name).exists():
            missing_files.append(file_name)
    
    if missing_dirs:
        print(f"❌ Missing directories: {missing_dirs}")
        return False
    
    if missing_files:
        print(f"❌ Missing files: {missing_files}")
        return False
    
    print("✅ All required directories and files present")
    return True

def main():
    """Run all structure validation tests."""
    print("=== Forex Trading Bot - Structure Validation ===")
    print(f"Project root: {project_root}")
    
    tests = [
        test_project_structure,
        test_imports,
        test_configuration,
        test_database_models,
        test_logging_setup
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
        print("🎉 All structure tests passed! Core system is properly set up.")
        return True
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)