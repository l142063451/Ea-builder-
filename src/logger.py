"""Structured logging system for the Forex Trading Bot."""

import json
import logging
import logging.handlers
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional

import structlog
from structlog.stdlib import LoggerFactory

from .config import settings

class JSONFormatter(logging.Formatter):
    """Custom JSON formatter for structured logging."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format the log record as JSON."""
        log_data = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "funcName": record.funcName,
            "lineno": record.lineno,
        }
        
        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        
        # Add any extra fields
        for key, value in record.__dict__.items():
            if key not in {"name", "msg", "args", "levelname", "levelno", "pathname", 
                          "filename", "module", "lineno", "funcName", "created", 
                          "msecs", "relativeCreated", "thread", "threadName", 
                          "processName", "process", "getMessage", "exc_info", 
                          "exc_text", "stack_info"}:
                log_data[key] = value
        
        return json.dumps(log_data, ensure_ascii=False)

def setup_logging() -> structlog.stdlib.BoundLogger:
    """Setup structured logging configuration."""
    
    # Create logs directory if it doesn't exist
    log_dir = Path(settings.logging.log_directory)
    log_dir.mkdir(exist_ok=True)
    
    # Configure standard logging
    logging.basicConfig(
        level=getattr(logging, settings.logging.log_level.upper()),
        format="%(message)s",
        handlers=[]
    )
    
    # Console handler with JSON formatting
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(JSONFormatter())
    
    # File handler with rotation
    file_handler = logging.handlers.TimedRotatingFileHandler(
        filename=log_dir / "trading_bot.log",
        when=settings.logging.log_rotation[0].upper(),  # D for daily, H for hourly
        interval=1,
        backupCount=settings.logging.log_retention_days,
        encoding="utf-8"
    )
    file_handler.setFormatter(JSONFormatter())
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)
    root_logger.setLevel(getattr(logging, settings.logging.log_level.upper()))
    
    # Configure structlog
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer(ensure_ascii=False)
        ],
        context_class=dict,
        logger_factory=LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    
    return structlog.get_logger("trading_bot")

class TradingLogger:
    """Enhanced logger with trading-specific methods."""
    
    def __init__(self, name: str = "trading_bot"):
        self.logger = structlog.get_logger(name)
    
    def log_trade_signal(
        self, 
        signal_type: str, 
        currency: str, 
        confidence: float, 
        strategy: str,
        **kwargs
    ) -> None:
        """Log a trading signal."""
        self.logger.info(
            "Trading signal generated",
            signal_type=signal_type,
            currency=currency,
            confidence=confidence,
            strategy=strategy,
            **kwargs
        )
    
    def log_trade_execution(
        self,
        trade_id: str,
        currency: str,
        trade_type: str,
        size: float,
        price: float,
        **kwargs
    ) -> None:
        """Log trade execution."""
        self.logger.info(
            "Trade executed",
            trade_id=trade_id,
            currency=currency,
            trade_type=trade_type,
            size=size,
            price=price,
            **kwargs
        )
    
    def log_performance_metrics(
        self,
        strategy: str,
        metrics: Dict[str, float],
        period: str = "daily"
    ) -> None:
        """Log performance metrics."""
        self.logger.info(
            "Performance metrics calculated",
            strategy=strategy,
            period=period,
            **metrics
        )
    
    def log_risk_event(
        self,
        event_type: str,
        severity: str,
        details: Dict[str, Any],
        action_taken: Optional[str] = None
    ) -> None:
        """Log risk management events."""
        self.logger.warning(
            "Risk event detected",
            event_type=event_type,
            severity=severity,
            action_taken=action_taken,
            **details
        )
    
    def log_data_quality(
        self,
        source: str,
        quality_score: float,
        issues: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log data quality metrics."""
        self.logger.info(
            "Data quality assessment",
            source=source,
            quality_score=quality_score,
            issues=issues or {}
        )
    
    def log_system_event(
        self,
        event_type: str,
        component: str,
        status: str,
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log system events."""
        self.logger.info(
            "System event",
            event_type=event_type,
            component=component,
            status=status,
            **(details or {})
        )

# Global logger instance
logger = setup_logging()
trading_logger = TradingLogger()