"""Enhanced data collection system with simulation fallback."""

import os
import json
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd

from .data_pipeline import DataCollector as BaseDataCollector
from .data_simulator import ForexDataSimulator
from .config import settings
from .logger import trading_logger


class AdvancedDataCollector(BaseDataCollector):
    """Enhanced data collector with simulation fallback and comprehensive data management."""
    
    def __init__(self, use_simulation: bool = False):
        """Initialize the enhanced data collector.
        
        Args:
            use_simulation: If True, use simulated data instead of live data
        """
        super().__init__()
        self.use_simulation = use_simulation
        self.simulator = ForexDataSimulator(seed=42) if use_simulation else None
        self.data_dir = Path("data")
        self.simulated_data_dir = Path("data/simulated")
        self.results_file = self.data_dir / "collection_results.json"
        
        # Create directories
        self.data_dir.mkdir(exist_ok=True)
        self.simulated_data_dir.mkdir(exist_ok=True)
        
        trading_logger.log_system_event(
            event_type="advanced_data_collector_init",
            component="AdvancedDataCollector",
            status="initialized",
            details={
                "use_simulation": use_simulation,
                "data_directory": str(self.data_dir.absolute()),
                "simulated_data_directory": str(self.simulated_data_dir.absolute())
            }
        )
    
    def collect_historical_data_comprehensive(self) -> Dict[str, Dict]:
        """Collect historical data with fallback to simulation if network fails."""
        
        trading_logger.logger.info("Starting comprehensive historical data collection")
        
        if self.use_simulation:
            return self._collect_simulated_data()
        else:
            # Try live data first, fall back to simulation if it fails
            live_results = self._attempt_live_data_collection()
            
            # Check if live data collection was successful for any pair
            successful_pairs = [
                symbol for symbol, result in live_results.items() 
                if result.get("status") == "success"
            ]
            
            if not successful_pairs:
                trading_logger.logger.warning(
                    "Live data collection failed for all pairs, falling back to simulation"
                )
                return self._collect_simulated_data()
            else:
                trading_logger.logger.info(
                    f"Live data collection successful for {len(successful_pairs)} pairs",
                    successful_pairs=successful_pairs
                )
                return live_results
    
    def _attempt_live_data_collection(self) -> Dict[str, Dict]:
        """Attempt to collect live data using the base collector."""
        try:
            return self.collect_all_currencies()
        except Exception as e:
            trading_logger.logger.error(
                "Live data collection completely failed",
                error=str(e)
            )
            return {symbol: {"status": "error", "error": str(e)} 
                   for symbol in settings.data.currency_pairs}
    
    def _collect_simulated_data(self) -> Dict[str, Dict]:
        """Collect simulated data and store it in the database."""
        
        trading_logger.logger.info("Generating and storing simulated forex data")
        
        results = {}
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(days=365 * settings.data.historical_data_years)
        
        for symbol in settings.data.currency_pairs:
            try:
                trading_logger.logger.info(f"Generating simulated data for {symbol}")
                
                # Generate simulated data
                data = self.simulator.generate_historical_data(
                    symbol=symbol,
                    start_date=start_date,
                    end_date=end_date,
                    timeframe="1m"
                )
                
                if data.empty:
                    results[symbol] = {
                        "status": "error",
                        "error": "No simulated data generated"
                    }
                    continue
                
                # Validate data quality
                quality_metrics = self.simulator._calculate_quality_metrics(data, symbol)
                
                # Store in database
                records_stored = self.store_price_data(data, symbol, "1m")
                
                # Save to CSV for inspection
                csv_file = self.simulated_data_dir / f"{symbol.replace('=', '_')}_1m_simulation.csv"
                data.to_csv(csv_file)
                
                results[symbol] = {
                    "status": "success",
                    "data_type": "simulated",
                    "records_generated": len(data),
                    "records_stored": records_stored,
                    "quality_metrics": quality_metrics,
                    "start_date": data.index[0].isoformat(),
                    "end_date": data.index[-1].isoformat(),
                    "csv_file": str(csv_file)
                }
                
                trading_logger.logger.info(
                    f"Successfully generated and stored simulated data for {symbol}",
                    records=records_stored,
                    quality_score=quality_metrics["quality_score"]
                )
                
            except Exception as e:
                results[symbol] = {
                    "status": "error",
                    "error": str(e)
                }
                
                trading_logger.logger.error(
                    f"Failed to generate simulated data for {symbol}",
                    error=str(e)
                )
        
        return results
    
    def save_collection_results(self, results: Dict[str, Dict]) -> None:
        """Save collection results to JSON file for analysis."""
        
        # Add metadata
        metadata = {
            "collection_timestamp": datetime.now(timezone.utc).isoformat(),
            "collection_method": "simulation" if self.use_simulation else "live_with_fallback",
            "total_pairs": len(results),
            "successful_pairs": len([r for r in results.values() if r.get("status") == "success"]),
            "total_records_stored": sum(r.get("records_stored", 0) for r in results.values()),
            "configuration": {
                "historical_years": settings.data.historical_data_years,
                "currency_pairs": settings.data.currency_pairs,
                "database_type": "sqlite" if "sqlite" in settings.database.url else "postgresql"
            }
        }
        
        output_data = {
            "metadata": metadata,
            "results": results
        }
        
        with open(self.results_file, 'w') as f:
            json.dump(output_data, f, indent=2, default=str)
        
        trading_logger.logger.info(
            "Collection results saved",
            file=str(self.results_file),
            successful_pairs=metadata["successful_pairs"],
            total_records=metadata["total_records_stored"]
        )
    
    def generate_data_quality_report(self) -> Dict:
        """Generate a comprehensive data quality report."""
        
        if not self.results_file.exists():
            return {"error": "No collection results found"}
        
        with open(self.results_file, 'r') as f:
            data = json.load(f)
        
        metadata = data.get("metadata", {})
        results = data.get("results", {})
        
        # Calculate aggregate quality metrics
        quality_scores = []
        completion_rates = []
        total_records = 0
        
        for symbol, result in results.items():
            if result.get("status") == "success":
                quality_metrics = result.get("quality_metrics", {})
                if quality_metrics:
                    quality_scores.append(quality_metrics.get("quality_score", 0))
                    completion_rates.append(quality_metrics.get("completeness", 0))
                    total_records += result.get("records_stored", 0)
        
        # Database statistics
        db_summary = self.get_data_summary()
        
        report = {
            "report_timestamp": datetime.now(timezone.utc).isoformat(),
            "collection_metadata": metadata,
            "aggregate_quality": {
                "average_quality_score": round(sum(quality_scores) / len(quality_scores), 2) if quality_scores else 0,
                "average_completion_rate": round(sum(completion_rates) / len(completion_rates), 2) if completion_rates else 0,
                "total_records_collected": total_records,
                "successful_pairs_count": len(quality_scores),
                "quality_threshold_met": all(score >= 95.0 for score in quality_scores)
            },
            "per_pair_results": results,
            "database_summary": db_summary,
            "quality_assessment": {
                "data_completeness": "EXCELLENT" if all(score >= 99.0 for score in completion_rates) else "GOOD" if all(score >= 95.0 for score in completion_rates) else "NEEDS_IMPROVEMENT",
                "overall_quality": "EXCELLENT" if all(score >= 99.0 for score in quality_scores) else "GOOD" if all(score >= 95.0 for score in quality_scores) else "NEEDS_IMPROVEMENT",
                "recommendation": self._get_quality_recommendation(quality_scores, completion_rates)
            }
        }
        
        return report
    
    def _get_quality_recommendation(self, quality_scores: List[float], completion_rates: List[float]) -> str:
        """Get quality improvement recommendations."""
        
        if not quality_scores:
            return "No data collected. Check network connectivity and data source availability."
        
        avg_quality = sum(quality_scores) / len(quality_scores)
        avg_completion = sum(completion_rates) / len(completion_rates)
        
        if avg_quality >= 99.0 and avg_completion >= 99.0:
            return "Excellent data quality. Ready for feature engineering phase."
        elif avg_quality >= 95.0 and avg_completion >= 95.0:
            return "Good data quality. Suitable for model development with minor optimizations."
        elif avg_quality >= 90.0 and avg_completion >= 90.0:
            return "Acceptable data quality. Consider data cleaning and validation improvements."
        else:
            return "Data quality below standards. Investigate data sources and collection methods."


def main():
    """Run comprehensive data collection with simulation fallback."""
    
    print("🚀 Starting Advanced Historical Data Collection")
    print("=" * 60)
    
    # Determine if we should use simulation (check for network connectivity issues)
    use_simulation = os.getenv("USE_SIMULATION", "false").lower() == "true"
    
    collector = AdvancedDataCollector(use_simulation=use_simulation)
    
    try:
        # Collect data
        results = collector.collect_historical_data_comprehensive()
        
        # Save results
        collector.save_collection_results(results)
        
        # Generate and display report
        report = collector.generate_data_quality_report()
        
        print("\n📊 Data Collection Summary")
        print("-" * 30)
        
        metadata = report.get("collection_metadata", {})
        aggregate = report.get("aggregate_quality", {})
        assessment = report.get("quality_assessment", {})
        
        print(f"Collection Method: {metadata.get('collection_method', 'unknown')}")
        print(f"Total Currency Pairs: {metadata.get('total_pairs', 0)}")
        print(f"Successful Collections: {metadata.get('successful_pairs', 0)}")
        print(f"Total Records Stored: {metadata.get('total_records_stored', 0):,}")
        print(f"Average Quality Score: {aggregate.get('average_quality_score', 0)}%")
        print(f"Data Completeness: {assessment.get('data_completeness', 'UNKNOWN')}")
        print(f"Overall Quality: {assessment.get('overall_quality', 'UNKNOWN')}")
        
        print("\n💡 Recommendation:")
        print(f"   {assessment.get('recommendation', 'No recommendation available')}")
        
        print("\n📋 Per-Pair Results:")
        for symbol, result in results.items():
            status_emoji = "✅" if result.get("status") == "success" else "❌"
            print(f"   {status_emoji} {symbol}: {result.get('status', 'unknown').upper()}")
            
            if result.get("status") == "success":
                records = result.get('records_stored', 0)
                quality = result.get('quality_metrics', {}).get('quality_score', 0)
                data_type = result.get('data_type', 'unknown')
                print(f"      Records: {records:,}, Quality: {quality}%, Type: {data_type}")
        
        # Stage completion assessment
        print("\n🎯 Stage 1 Completion Assessment:")
        threshold_met = aggregate.get('quality_threshold_met', False)
        min_records = 100000  # Minimum records for meaningful analysis
        total_records = aggregate.get('total_records_collected', 0)
        
        criteria = [
            ("Data Quality >95%", threshold_met),
            (f"Minimum Records ({min_records:,})", total_records >= min_records),
            ("Database Operational", len(report.get('database_summary', {})) > 0),
            ("All Currency Pairs", aggregate.get('successful_pairs_count', 0) >= 4)
        ]
        
        all_met = all(met for _, met in criteria)
        
        for criterion, met in criteria:
            status = "✅" if met else "❌"
            print(f"   {status} {criterion}")
        
        if all_met:
            print("\n🎉 Stage 1 (Data Collection) COMPLETED!")
            print("   Ready to advance to Stage 2 (Feature Engineering)")
        else:
            print("\n⚠️  Stage 1 completion criteria not fully met")
            print("   Continue data collection improvements")
        
        print(f"\n📁 Detailed results saved to: {collector.results_file}")
        print("=" * 60)
        
        return results
        
    except Exception as e:
        trading_logger.logger.error("Data collection failed", error=str(e))
        print(f"❌ Data collection failed: {e}")
        raise
    finally:
        collector.close()


if __name__ == "__main__":
    main()