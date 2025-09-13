# Forex Autonomous Trading Bot

An advanced autonomous Forex trading system that combines artificial intelligence, machine learning, and robust risk management to achieve consistent profitability while maintaining capital preservation as the highest priority.

## 🎯 Vision

To develop the world's most sophisticated, autonomous Forex trading system that operates 24/7 across multiple currency pairs with minimal human intervention, learning and adapting from market conditions using advanced AI and machine learning.

## 📊 Performance Targets

- **Predictive Accuracy**: ≥90% accuracy on entry/exit signals
- **Maximum Drawdown**: <10% at any point in time
- **Sharpe Ratio**: >2.0 on annual basis
- **Win Rate**: >65% of all trades
- **Risk-Reward Ratio**: Minimum 1:2
- **Monthly Return**: Target 5-15% with <5% monthly drawdown

## 🏗️ Project Structure

```
├── src/                    # Core source code
│   ├── __init__.py
│   ├── config.py           # Configuration management
│   ├── database.py         # Database models and schema
│   ├── logger.py           # Structured logging system
│   └── data_pipeline.py    # Data collection pipeline
├── tests/                  # Test suite
│   ├── test_structure.py   # Structure validation tests
│   └── test_data_collection.py
├── examples/               # Usage examples
│   └── basic_usage.py
├── config/                 # Configuration files
│   └── .env.example
├── data/                   # Data storage
├── logs/                   # Application logs
├── requirements.txt        # Python dependencies
└── pyproject.toml         # Project configuration
```

## 🚀 Getting Started

### Prerequisites

- Python 3.9+
- PostgreSQL (for production) or SQLite (for development)

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd Ea-builder-
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure environment variables:
```bash
cp config/.env.example .env
# Edit .env with your settings
```

### Basic Usage

```python
# Set up environment for SQLite (development)
import os
os.environ['DATABASE_URL'] = 'sqlite:///forex_bot.db'

from src.config import settings
from src.database import Base, Currency
from sqlalchemy import create_engine

# Initialize database
engine = create_engine(settings.database.url)
Base.metadata.create_all(bind=engine)
```

## 🧪 Testing

Run the structure validation tests:

```bash
python tests/test_structure.py
```

This validates that all core modules can be imported and basic functionality works.

## 📈 Current Status

**Stage**: Data Collection and Preparation (In Progress)

### ✅ Completed
- Project structure setup
- Configuration management system
- Database models and schema
- Structured logging system
- Basic data collection pipeline
- Test infrastructure

### 🔄 In Progress
- Historical data collection from yfinance
- Data quality validation
- Database integration testing

### ⏳ Planned
- Feature engineering (technical indicators)
- Machine learning models
- Backtesting framework
- Paper trading system
- Risk management system
- Live deployment

## 🛡️ Safety Features

- **Demo Mode Default**: System defaults to paper trading
- **Human Approval Required**: Live trading requires explicit human approval
- **Circuit Breakers**: Automatic trading halts on risk thresholds
- **Comprehensive Logging**: Complete audit trail for all decisions

## 📚 Documentation

- [Requirements and Goals](Requirements_and_Goals.md) - Complete project specifications
- [Status](Status.md) - Current progress and next steps
- [Copilot Instructions](Copilot_Instructions.md) - Development guidelines

## 🤝 Contributing

This project follows a strict development sequence:
1. Data Stage → 2. Feature Engineering → 3. Model Development → 4. Backtesting → 
5. Memory System → 6. Paper Trading → 7. Risk Management → 8. Monitoring → 
9. Deployment → 10. Continuous Learning

Each stage must meet defined performance criteria before advancement.

## ⚠️ Disclaimer

This trading bot is for educational and research purposes. Trading involves significant risk and past performance does not guarantee future results. Always trade responsibly and never risk more than you can afford to lose.

## 📄 License

MIT License - see LICENSE file for details.