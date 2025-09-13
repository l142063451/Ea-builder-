"""Data validation and testing system for the collected forex data."""

import sqlite3
import pandas as pd
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple
import json

from src.config import settings
from src.logger import trading_logger


class DataValidator:
    """Validates collected data and provides comprehensive analysis."""
    
    def __init__(self, db_path: str = "data/forex_trading.db"):
        """Initialize the data validator.
        
        Args:
            db_path: Path to the SQLite database
        """
        self.db_path = Path(db_path)
        self.connection = None
        
        if self.db_path.exists():
            self.connection = sqlite3.connect(str(self.db_path))
            self.connection.row_factory = sqlite3.Row
    
    def validate_data_integrity(self) -> Dict:
        """Perform comprehensive data integrity validation."""
        if not self.connection:
            return {"error": "Database not available"}
        
        validation_results = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "database_path": str(self.db_path.absolute()),
            "database_size_mb": self.db_path.stat().st_size / (1024 * 1024) if self.db_path.exists() else 0,
            "currencies": {},
            "overall_stats": {},
            "data_quality": {},
            "validation_status": "PASS"
        }
        
        try:
            # Get currency information
            currencies = self._get_currencies()
            
            total_records = 0
            quality_scores = []
            
            for currency in currencies:
                currency_stats = self._validate_currency_data(currency)
                validation_results["currencies"][currency['symbol']] = currency_stats
                
                total_records += currency_stats.get("record_count", 0)
                if currency_stats.get("quality_score"):
                    quality_scores.append(currency_stats["quality_score"])
            
            # Overall statistics
            validation_results["overall_stats"] = {
                "total_currencies": len(currencies),
                "total_records": total_records,
                "average_quality_score": sum(quality_scores) / len(quality_scores) if quality_scores else 0,
                "data_completeness": len([c for c in currencies if validation_results["currencies"][c['symbol']].get("record_count", 0) > 0]) / len(currencies) * 100 if currencies else 0
            }
            
            # Data quality assessment
            avg_quality = validation_results["overall_stats"]["average_quality_score"]
            data_complete = validation_results["overall_stats"]["data_completeness"]
            
            if avg_quality >= 95 and data_complete >= 95 and total_records >= 100000:
                validation_results["validation_status"] = "PASS"
                validation_results["data_quality"]["assessment"] = "EXCELLENT"
                validation_results["data_quality"]["ready_for_next_stage"] = True
            elif avg_quality >= 90 and data_complete >= 90 and total_records >= 50000:
                validation_results["validation_status"] = "PASS_WITH_WARNINGS"
                validation_results["data_quality"]["assessment"] = "GOOD"
                validation_results["data_quality"]["ready_for_next_stage"] = True
            else:
                validation_results["validation_status"] = "FAIL"
                validation_results["data_quality"]["assessment"] = "NEEDS_IMPROVEMENT"
                validation_results["data_quality"]["ready_for_next_stage"] = False
            
            trading_logger.logger.info(
                "Data validation completed",
                status=validation_results["validation_status"],
                total_records=total_records,
                quality_score=avg_quality
            )
            
        except Exception as e:
            validation_results["error"] = str(e)
            validation_results["validation_status"] = "ERROR"
            
            trading_logger.logger.error(
                "Data validation failed",
                error=str(e)
            )
        
        return validation_results
    
    def _get_currencies(self) -> List[Dict]:
        """Get all currency pairs from the database."""
        cursor = self.connection.cursor()
        cursor.execute("SELECT * FROM currencies WHERE is_active = 1")
        return [dict(row) for row in cursor.fetchall()]
    
    def _validate_currency_data(self, currency: Dict) -> Dict:
        """Validate data for a specific currency pair."""
        cursor = self.connection.cursor()
        symbol = currency['symbol']
        currency_id = currency['id']
        
        # Basic statistics
        cursor.execute("""
            SELECT 
                COUNT(*) as record_count,
                MIN(timestamp) as first_timestamp,
                MAX(timestamp) as last_timestamp,
                MIN(open_price) as min_price,
                MAX(high_price) as max_price,
                AVG(close_price) as avg_price
            FROM price_data 
            WHERE currency_id = ? AND timeframe = '1m'
        """, (currency_id,))
        
        stats = dict(cursor.fetchone())
        
        if stats['record_count'] == 0:
            return {
                "record_count": 0,
                "quality_score": 0,
                "issues": ["No data found"]
            }
        
        # Data quality checks
        issues = []
        
        # Check for price consistency (OHLC relationships)
        cursor.execute("""
            SELECT COUNT(*) as invalid_ohlc
            FROM price_data 
            WHERE currency_id = ? AND timeframe = '1m'
            AND (low_price > open_price OR low_price > close_price OR
                 high_price < open_price OR high_price < close_price OR
                 high_price < low_price)
        """, (currency_id,))
        
        invalid_ohlc = cursor.fetchone()['invalid_ohlc']
        if invalid_ohlc > 0:
            issues.append(f"{invalid_ohlc} records with invalid OHLC relationships")
        
        # Check for missing data gaps (more than 5 minutes between consecutive records)
        cursor.execute("""
            WITH time_gaps AS (
                SELECT 
                    timestamp,
                    LAG(timestamp) OVER (ORDER BY timestamp) as prev_timestamp,
                    (julianday(timestamp) - julianday(LAG(timestamp) OVER (ORDER BY timestamp))) * 1440 as gap_minutes
                FROM price_data 
                WHERE currency_id = ? AND timeframe = '1m'
                ORDER BY timestamp
            )
            SELECT COUNT(*) as large_gaps
            FROM time_gaps 
            WHERE gap_minutes > 5
        """, (currency_id,))
        
        large_gaps = cursor.fetchone()['large_gaps']
        if large_gaps > stats['record_count'] * 0.01:  # More than 1% gaps
            issues.append(f"{large_gaps} significant time gaps detected")
        
        # Check for unrealistic price movements (more than 10% change)
        cursor.execute("""
            WITH price_changes AS (
                SELECT 
                    close_price,
                    LAG(close_price) OVER (ORDER BY timestamp) as prev_close,
                    ABS(close_price - LAG(close_price) OVER (ORDER BY timestamp)) / LAG(close_price) OVER (ORDER BY timestamp) as pct_change
                FROM price_data 
                WHERE currency_id = ? AND timeframe = '1m'
                ORDER BY timestamp
            )
            SELECT COUNT(*) as extreme_movements
            FROM price_changes 
            WHERE pct_change > 0.1
        """, (currency_id,))
        
        extreme_movements = cursor.fetchone()['extreme_movements']
        if extreme_movements > 0:
            issues.append(f"{extreme_movements} extreme price movements (>10%)")
        
        # Calculate quality score
        ohlc_quality = 1 - (invalid_ohlc / stats['record_count'])
        gap_quality = 1 - (large_gaps / max(stats['record_count'], 1))
        movement_quality = 1 - (extreme_movements / max(stats['record_count'], 1))
        
        quality_score = (ohlc_quality * 0.4 + gap_quality * 0.4 + movement_quality * 0.2) * 100
        
        return {
            "record_count": stats['record_count'],
            "date_range": {
                "start": stats['first_timestamp'],
                "end": stats['last_timestamp']
            },
            "price_range": {
                "min": float(stats['min_price']) if stats['min_price'] else None,
                "max": float(stats['max_price']) if stats['max_price'] else None,
                "avg": float(stats['avg_price']) if stats['avg_price'] else None
            },
            "quality_score": round(quality_score, 2),
            "data_quality_metrics": {
                "ohlc_consistency": round(ohlc_quality * 100, 2),
                "time_continuity": round(gap_quality * 100, 2),
                "price_realism": round(movement_quality * 100, 2)
            },
            "issues": issues,
            "status": "PASS" if quality_score >= 95 else "WARNING" if quality_score >= 90 else "FAIL"
        }
    
    def generate_summary_report(self, save_to_file: bool = True) -> Dict:
        """Generate a comprehensive summary report."""
        validation_results = self.validate_data_integrity()
        
        # Create summary report
        report = {
            "report_timestamp": datetime.now(timezone.utc).isoformat(),
            "data_collection_summary": {
                "database_size_mb": validation_results.get("database_size_mb", 0),
                "total_records": validation_results.get("overall_stats", {}).get("total_records", 0),
                "total_currencies": validation_results.get("overall_stats", {}).get("total_currencies", 0),
                "validation_status": validation_results.get("validation_status", "UNKNOWN")
            },
            "quality_assessment": {
                "average_quality_score": validation_results.get("overall_stats", {}).get("average_quality_score", 0),
                "data_completeness": validation_results.get("overall_stats", {}).get("data_completeness", 0),
                "overall_assessment": validation_results.get("data_quality", {}).get("assessment", "UNKNOWN"),
                "ready_for_next_stage": validation_results.get("data_quality", {}).get("ready_for_next_stage", False)
            },
            "currency_details": validation_results.get("currencies", {}),
            "recommendations": self._generate_recommendations(validation_results)
        }
        
        if save_to_file:
            report_file = Path("data") / "data_validation_report.json"
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            trading_logger.logger.info(
                "Data validation report saved",
                file=str(report_file),
                status=report["data_collection_summary"]["validation_status"]
            )
        
        return report
    
    def _generate_recommendations(self, validation_results: Dict) -> List[str]:
        """Generate recommendations based on validation results."""
        recommendations = []
        
        overall_stats = validation_results.get("overall_stats", {})
        total_records = overall_stats.get("total_records", 0)
        avg_quality = overall_stats.get("average_quality_score", 0)
        data_complete = overall_stats.get("data_completeness", 0)
        
        if total_records < 100000:
            recommendations.append("Increase data collection volume - minimum 100,000 records needed for reliable analysis")
        
        if avg_quality < 95:
            recommendations.append("Improve data quality - target >95% quality score")
        
        if data_complete < 95:
            recommendations.append("Complete data collection for all currency pairs")
        
        if validation_results.get("validation_status") == "PASS":
            recommendations.append("✅ Data collection phase complete - Ready to advance to Feature Engineering stage")
            recommendations.append("🎯 Next: Implement technical indicators (SMA, EMA, RSI, MACD, Bollinger Bands)")
        elif validation_results.get("validation_status") == "PASS_WITH_WARNINGS":
            recommendations.append("⚠️ Data acceptable but could be improved before advancing")
            recommendations.append("Consider addressing quality issues or proceed with caution")
        else:
            recommendations.append("❌ Data quality insufficient - Continue improving data collection")
        
        return recommendations
    
    def close(self):
        """Close database connection."""
        if self.connection:
            self.connection.close()


def main():
    """Run data validation and generate report."""
    print("🔍 Starting Data Validation and Analysis")
    print("=" * 50)
    
    validator = DataValidator()
    
    try:
        # Generate comprehensive report
        report = validator.generate_summary_report()
        
        # Display results
        summary = report["data_collection_summary"]
        quality = report["quality_assessment"]
        
        print(f"\n📊 Data Collection Summary:")
        print(f"   Database Size: {summary['database_size_mb']:.1f} MB")
        print(f"   Total Records: {summary['total_records']:,}")
        print(f"   Currency Pairs: {summary['total_currencies']}")
        print(f"   Validation Status: {summary['validation_status']}")
        
        print(f"\n🎯 Quality Assessment:")
        print(f"   Average Quality Score: {quality['average_quality_score']:.1f}%")
        print(f"   Data Completeness: {quality['data_completeness']:.1f}%")
        print(f"   Overall Assessment: {quality['overall_assessment']}")
        print(f"   Ready for Next Stage: {'✅ Yes' if quality['ready_for_next_stage'] else '❌ No'}")
        
        print(f"\n📋 Per-Currency Results:")
        for symbol, details in report["currency_details"].items():
            status_emoji = "✅" if details['status'] == 'PASS' else "⚠️" if details['status'] == 'WARNING' else "❌"
            print(f"   {status_emoji} {symbol}: {details['record_count']:,} records, {details['quality_score']:.1f}% quality")
        
        print(f"\n💡 Recommendations:")
        for rec in report["recommendations"]:
            print(f"   • {rec}")
        
        print(f"\n📁 Full report saved to: data/data_validation_report.json")
        print("=" * 50)
        
        return report
        
    except Exception as e:
        print(f"❌ Validation failed: {e}")
        trading_logger.logger.error("Data validation failed", error=str(e))
        raise
    finally:
        validator.close()


if __name__ == "__main__":
    main()