"""Basic usage example for the Forex Trading Bot data collection."""

import os
import sys
from pathlib import Path

# Set SQLite database URL before importing anything
os.environ['DATABASE_URL'] = 'sqlite:///example_forex.db'

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def main():
    """Demonstrate basic functionality."""
    print("=== Forex Trading Bot - Basic Example ===")
    
    # Import after setting environment
    from src.config import settings
    from src.database import Currency, Base
    from src.logger import trading_logger
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    
    print(f"Configuration loaded:")
    print(f"  Demo mode: {settings.trading.demo_mode}")
    print(f"  Target accuracy: {settings.performance.target_accuracy}")
    print(f"  Currency pairs: {settings.data.currency_pairs}")
    
    # Set up database
    engine = create_engine(settings.database.url, echo=False)
    Base.metadata.create_all(bind=engine)
    SessionLocal = sessionmaker(bind=engine)
    db = SessionLocal()
    
    # Create example currencies
    for symbol in settings.data.currency_pairs:
        existing = db.query(Currency).filter(Currency.symbol == symbol).first()
        if not existing:
            base_currency = symbol.split('USD')[0] if 'USD' in symbol else symbol[:3]
            quote_currency = 'USD' if 'USD' in symbol else symbol[3:6]
            
            currency = Currency(
                symbol=symbol,
                name=f"{base_currency}/{quote_currency}",
                base_currency=base_currency,
                quote_currency=quote_currency
            )
            db.add(currency)
            print(f"Added currency: {symbol}")
    
    db.commit()
    
    # Show summary
    currency_count = db.query(Currency).count()
    print(f"\nDatabase summary:")
    print(f"  Total currencies: {currency_count}")
    
    # Log completion
    trading_logger.log_system_event(
        event_type="example_completion",
        component="basic_example",
        status="completed",
        details={"currencies_created": currency_count}
    )
    
    db.close()
    print("\n✅ Basic example completed successfully!")

if __name__ == "__main__":
    main()