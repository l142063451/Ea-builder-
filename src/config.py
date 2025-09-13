"""Configuration management for the Forex Trading Bot."""

import os
from pathlib import Path
from typing import Dict, Any, Optional
from pydantic_settings import BaseSettings
from pydantic import Field, field_validator
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class DatabaseConfig(BaseSettings):
    """Database configuration settings."""
    
    host: str = Field(default="localhost", env="DATABASE_HOST")
    port: int = Field(default=5432, env="DATABASE_PORT")
    name: str = Field(default="forex_trading_bot", env="DATABASE_NAME")
    user: str = Field(default="trading_user", env="DATABASE_USER")
    password: str = Field(default="", env="DATABASE_PASSWORD")
    url: Optional[str] = Field(default=None, env="DATABASE_URL")
    
    @field_validator("url", mode="before")
    @classmethod
    def assemble_db_url(cls, v: Optional[str], info) -> str:
        if isinstance(v, str) and v:
            return v
        values = info.data if hasattr(info, 'data') else {}
        return f"postgresql://{values.get('user', 'trading_user')}:{values.get('password', '')}@{values.get('host', 'localhost')}:{values.get('port', 5432)}/{values.get('name', 'forex_trading_bot')}"

class TradingConfig(BaseSettings):
    """Trading configuration settings."""
    
    demo_mode: bool = Field(default=True, env="DEMO_MODE")
    live_trading_enabled: bool = Field(default=False, env="LIVE_TRADING_ENABLED")
    max_daily_loss_percent: float = Field(default=2.0, env="MAX_DAILY_LOSS_PERCENT")
    max_position_size_percent: float = Field(default=5.0, env="MAX_POSITION_SIZE_PERCENT")
    max_drawdown_percent: float = Field(default=10.0, env="MAX_DRAWDOWN_PERCENT")
    max_concurrent_trades: int = Field(default=5, env="MAX_CONCURRENT_TRADES")
    execution_latency_ms: int = Field(default=50, env="EXECUTION_LATENCY_MS")

class DataConfig(BaseSettings):
    """Data source configuration settings."""
    
    yfinance_timeout: int = Field(default=30, env="YFINANCE_TIMEOUT")
    data_update_interval: int = Field(default=60, env="DATA_UPDATE_INTERVAL")
    historical_data_years: int = Field(default=5, env="HISTORICAL_DATA_YEARS")
    currency_pairs: list = Field(default=["EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X"])

class RiskConfig(BaseSettings):
    """Risk management configuration settings."""
    
    kelly_fraction_limit: float = Field(default=0.25, env="KELLY_FRACTION_LIMIT")
    volatility_lookback_days: int = Field(default=20, env="VOLATILITY_LOOKBACK_DAYS")
    stop_loss_atr_multiplier: float = Field(default=2.0, env="STOP_LOSS_ATR_MULTIPLIER")
    take_profit_multiplier: float = Field(default=2.0, env="TAKE_PROFIT_MULTIPLIER")

class PerformanceConfig(BaseSettings):
    """Performance target configuration settings."""
    
    target_accuracy: float = Field(default=0.90, env="TARGET_ACCURACY")
    target_sharpe_ratio: float = Field(default=2.0, env="TARGET_SHARPE_RATIO")
    target_win_rate: float = Field(default=0.65, env="TARGET_WIN_RATE")
    target_profit_factor: float = Field(default=1.5, env="TARGET_PROFIT_FACTOR")
    system_uptime_target: float = Field(default=99.5, env="SYSTEM_UPTIME_TARGET")

class LoggingConfig(BaseSettings):
    """Logging configuration settings."""
    
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_rotation: str = Field(default="daily", env="LOG_ROTATION")
    log_retention_days: int = Field(default=30, env="LOG_RETENTION_DAYS")
    log_directory: Path = Field(default=Path("logs"))

class Settings:
    """Main settings class that combines all configuration sections."""
    
    def __init__(self):
        self.database = DatabaseConfig()
        self.trading = TradingConfig()
        self.data = DataConfig()
        self.risk = RiskConfig()
        self.performance = PerformanceConfig()
        self.logging = LoggingConfig()
        
        # Create necessary directories
        self.logging.log_directory.mkdir(exist_ok=True)
        Path("data").mkdir(exist_ok=True)
        Path("models").mkdir(exist_ok=True)

# Global settings instance
settings = Settings()