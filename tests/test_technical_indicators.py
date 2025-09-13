"""
Test suite for Technical Indicators module.

This module provides comprehensive testing for all technical indicators
to ensure accuracy, performance, and reliability.
"""

import numpy as np
import pandas as pd
from unittest import TestCase
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from technical_indicators import TechnicalIndicators, IndicatorValidator, create_sample_data, IndicatorResult


class TestTechnicalIndicators(TestCase):
    """Test cases for TechnicalIndicators class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.calc = TechnicalIndicators()
        self.sample_data = create_sample_data(100)
        
        # Create test data with known expected results
        self.simple_data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    
    def test_input_validation(self):
        """Test input validation functionality."""
        # Test valid inputs
        valid_array = np.array([1, 2, 3, 4, 5])
        result = self.calc.validate_input(valid_array, 3)
        np.testing.assert_array_equal(result, valid_array)
        
        # Test insufficient data
        with self.assertRaises(ValueError):
            self.calc.validate_input([1, 2], 5)
        
        # Test None input
        with self.assertRaises(ValueError):
            self.calc.validate_input(None, 1)
        
        # Test list input conversion
        list_data = [1, 2, 3, 4, 5]
        result = self.calc.validate_input(list_data, 3)
        np.testing.assert_array_equal(result, np.array(list_data, dtype=np.float64))
    
    def test_sma_calculation(self):
        """Test Simple Moving Average calculation."""
        # Test with simple known data
        sma_result = self.calc.sma(self.simple_data, 3)
        
        # Verify result structure
        self.assertIsInstance(sma_result, IndicatorResult)
        self.assertEqual(sma_result.name, "SMA")
        self.assertEqual(sma_result.parameters["period"], 3)
        
        # Verify calculations
        expected_values = [np.nan, np.nan, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
        np.testing.assert_array_almost_equal(sma_result.values, expected_values)
        
        # Test with sample forex data
        sma_20 = self.calc.sma(self.sample_data["close"], 20)
        self.assertEqual(len(sma_20.values), len(self.sample_data["close"]))
        self.assertTrue(np.isnan(sma_20.values[18]))  # Should be NaN before period-1
        self.assertFalse(np.isnan(sma_20.values[19]))  # Should have value at period-1
    
    def test_ema_calculation(self):
        """Test Exponential Moving Average calculation."""
        ema_result = self.calc.ema(self.simple_data, 3)
        
        # Verify result structure
        self.assertIsInstance(ema_result, IndicatorResult)
        self.assertEqual(ema_result.name, "EMA")
        self.assertEqual(ema_result.parameters["period"], 3)
        
        # First EMA should equal SMA
        first_sma = np.mean(self.simple_data[:3])
        self.assertAlmostEqual(ema_result.values[2], first_sma, places=6)
        
        # Test multiplier calculation
        expected_multiplier = 2.0 / (3 + 1)
        self.assertEqual(ema_result.parameters["multiplier"], expected_multiplier)
        
        # Verify EMA values are not NaN after initialization
        self.assertFalse(np.isnan(ema_result.values[3]))
        self.assertFalse(np.isnan(ema_result.values[-1]))
    
    def test_rsi_calculation(self):
        """Test Relative Strength Index calculation."""
        # Create data with known trend (increasing prices)
        increasing_data = np.array([100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 
                                   110, 111, 112, 113, 114, 115])
        
        rsi_result = self.calc.rsi(increasing_data, 14)
        
        # Verify result structure
        self.assertIsInstance(rsi_result, IndicatorResult)
        self.assertEqual(rsi_result.name, "RSI")
        self.assertEqual(rsi_result.parameters["period"], 14)
        
        # RSI should be close to 100 for continuously increasing prices
        self.assertGreater(rsi_result.values[-1], 90)
        self.assertLessEqual(rsi_result.values[-1], 100)
        
        # Test with sample data
        rsi_14 = self.calc.rsi(self.sample_data["close"], 14)
        valid_values = rsi_14.values[~np.isnan(rsi_14.values)]
        
        # All RSI values should be between 0 and 100
        self.assertTrue(np.all(valid_values >= 0))
        self.assertTrue(np.all(valid_values <= 100))
    
    def test_macd_calculation(self):
        """Test MACD calculation."""
        macd_result = self.calc.macd(self.sample_data["close"], 12, 26, 9)
        
        # Verify result structure
        self.assertIsInstance(macd_result, IndicatorResult)
        self.assertEqual(macd_result.name, "MACD")
        self.assertEqual(macd_result.parameters["fast_period"], 12)
        self.assertEqual(macd_result.parameters["slow_period"], 26)
        self.assertEqual(macd_result.parameters["signal_period"], 9)
        
        # Verify three columns (MACD, Signal, Histogram)
        self.assertEqual(macd_result.values.shape[1], 3)
        self.assertEqual(len(macd_result.metadata["columns"]), 3)
        
        # MACD line should have values after slow period
        macd_line = macd_result.values[:, 0]
        self.assertTrue(np.isnan(macd_line[24]))  # Before slow_period - 1
        self.assertFalse(np.isnan(macd_line[25]))  # At slow_period - 1
        
        # Histogram should equal MACD - Signal (where both exist)
        macd_vals = macd_result.values[:, 0]
        signal_vals = macd_result.values[:, 1]
        histogram_vals = macd_result.values[:, 2]
        
        valid_mask = ~(np.isnan(macd_vals) | np.isnan(signal_vals))
        calculated_histogram = macd_vals[valid_mask] - signal_vals[valid_mask]
        actual_histogram = histogram_vals[valid_mask]
        
        np.testing.assert_array_almost_equal(calculated_histogram, actual_histogram, decimal=10)
    
    def test_bollinger_bands_calculation(self):
        """Test Bollinger Bands calculation."""
        bb_result = self.calc.bollinger_bands(self.sample_data["close"], 20, 2.0)
        
        # Verify result structure
        self.assertIsInstance(bb_result, IndicatorResult)
        self.assertEqual(bb_result.name, "Bollinger_Bands")
        self.assertEqual(bb_result.parameters["period"], 20)
        self.assertEqual(bb_result.parameters["std_dev"], 2.0)
        
        # Verify three bands
        self.assertEqual(bb_result.values.shape[1], 3)
        upper_band = bb_result.values[:, 0]
        middle_band = bb_result.values[:, 1]
        lower_band = bb_result.values[:, 2]
        
        # Upper band should be > middle band > lower band (where valid)
        valid_idx = ~np.isnan(middle_band)
        valid_indices = np.where(valid_idx)[0]
        
        for idx in valid_indices:
            self.assertGreater(upper_band[idx], middle_band[idx])
            self.assertGreater(middle_band[idx], lower_band[idx])
        
        # Middle band should equal SMA
        sma_result = self.calc.sma(self.sample_data["close"], 20)
        np.testing.assert_array_almost_equal(middle_band, sma_result.values, decimal=10)
    
    def test_atr_calculation(self):
        """Test Average True Range calculation."""
        atr_result = self.calc.atr(
            self.sample_data["high"], 
            self.sample_data["low"], 
            self.sample_data["close"], 
            14
        )
        
        # Verify result structure
        self.assertIsInstance(atr_result, IndicatorResult)
        self.assertEqual(atr_result.name, "ATR")
        self.assertEqual(atr_result.parameters["period"], 14)
        
        # ATR values should be positive
        valid_values = atr_result.values[~np.isnan(atr_result.values)]
        self.assertTrue(np.all(valid_values > 0))
        
        # Test array length validation
        with self.assertRaises(ValueError):
            self.calc.atr([1, 2, 3], [1, 2], [1, 2, 3], 14)
    
    def test_stochastic_calculation(self):
        """Test Stochastic Oscillator calculation."""
        stoch_result = self.calc.stochastic(
            self.sample_data["high"], 
            self.sample_data["low"], 
            self.sample_data["close"], 
            14, 3
        )
        
        # Verify result structure
        self.assertIsInstance(stoch_result, IndicatorResult)
        self.assertEqual(stoch_result.name, "Stochastic")
        self.assertEqual(stoch_result.parameters["k_period"], 14)
        self.assertEqual(stoch_result.parameters["d_period"], 3)
        
        # Verify two columns (%K and %D)
        self.assertEqual(stoch_result.values.shape[1], 2)
        k_values = stoch_result.values[:, 0]
        d_values = stoch_result.values[:, 1]
        
        # %K and %D should be between 0 and 100
        valid_k = k_values[~np.isnan(k_values)]
        valid_d = d_values[~np.isnan(d_values)]
        
        self.assertTrue(np.all(valid_k >= 0))
        self.assertTrue(np.all(valid_k <= 100))
        self.assertTrue(np.all(valid_d >= 0))
        self.assertTrue(np.all(valid_d <= 100))
    
    def test_williams_r_calculation(self):
        """Test Williams %R calculation."""
        wr_result = self.calc.williams_r(
            self.sample_data["high"], 
            self.sample_data["low"], 
            self.sample_data["close"], 
            14
        )
        
        # Verify result structure
        self.assertIsInstance(wr_result, IndicatorResult)
        self.assertEqual(wr_result.name, "Williams_R")
        self.assertEqual(wr_result.parameters["period"], 14)
        
        # Williams %R should be between -100 and 0
        valid_values = wr_result.values[~np.isnan(wr_result.values)]
        self.assertTrue(np.all(valid_values >= -100))
        self.assertTrue(np.all(valid_values <= 0))
    
    def test_cci_calculation(self):
        """Test Commodity Channel Index calculation."""
        cci_result = self.calc.cci(
            self.sample_data["high"], 
            self.sample_data["low"], 
            self.sample_data["close"], 
            20
        )
        
        # Verify result structure
        self.assertIsInstance(cci_result, IndicatorResult)
        self.assertEqual(cci_result.name, "CCI")
        self.assertEqual(cci_result.parameters["period"], 20)
        self.assertEqual(cci_result.parameters["constant"], 0.015)
        
        # CCI values should be finite
        valid_values = cci_result.values[~np.isnan(cci_result.values)]
        self.assertTrue(np.all(np.isfinite(valid_values)))


class TestIndicatorValidator(TestCase):
    """Test cases for IndicatorValidator class."""
    
    def test_validate_indicator_accuracy(self):
        """Test indicator accuracy validation."""
        # Perfect match case
        calculated = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        expected = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        
        result = IndicatorValidator.validate_indicator_accuracy(calculated, expected)
        
        self.assertTrue(result["valid"])
        self.assertEqual(result["accuracy_percent"], 100.0)
        self.assertEqual(result["max_difference"], 0.0)
        
        # Small difference case
        calculated = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        expected = np.array([1.0001, 2.0001, 3.0001, 4.0001, 5.0001])
        
        result = IndicatorValidator.validate_indicator_accuracy(calculated, expected, tolerance=1e-3)
        
        self.assertTrue(result["valid"])
        self.assertAlmostEqual(result["max_difference"], 0.0001, places=5)
        
        # Length mismatch case
        calculated = np.array([1.0, 2.0, 3.0])
        expected = np.array([1.0, 2.0, 3.0, 4.0])
        
        result = IndicatorValidator.validate_indicator_accuracy(calculated, expected)
        
        self.assertFalse(result["valid"])
        self.assertIn("Length mismatch", result["error"])
    
    def test_generate_validation_report(self):
        """Test validation report generation."""
        calc = TechnicalIndicators()
        sample_data = create_sample_data(50)
        
        # Create some indicator results
        indicators = {
            "SMA_20": calc.sma(sample_data["close"], 20),
            "EMA_12": calc.ema(sample_data["close"], 12),
            "RSI_14": calc.rsi(sample_data["close"], 14)
        }
        
        report = IndicatorValidator.generate_validation_report(indicators)
        
        # Verify report structure
        self.assertEqual(report["total_indicators"], 3)
        self.assertIn("validation_timestamp", report)
        self.assertIn("indicators", report)
        
        # Verify each indicator report
        for name in ["SMA_20", "EMA_12", "RSI_14"]:
            self.assertIn(name, report["indicators"])
            indicator_report = report["indicators"][name]
            
            self.assertIn("name", indicator_report)
            self.assertIn("parameters", indicator_report)
            self.assertIn("total_points", indicator_report)
            self.assertIn("valid_points", indicator_report)
            self.assertIn("completeness_percent", indicator_report)


class TestSampleDataGeneration(TestCase):
    """Test cases for sample data generation."""
    
    def test_create_sample_data(self):
        """Test sample data creation."""
        length = 100
        start_price = 1.2000
        
        data = create_sample_data(length, start_price)
        
        # Verify structure
        self.assertIn("open", data)
        self.assertIn("high", data)
        self.assertIn("low", data)
        self.assertIn("close", data)
        
        # Verify lengths
        for key, values in data.items():
            self.assertEqual(len(values), length)
        
        # Verify OHLC relationships
        for i in range(length):
            self.assertGreaterEqual(data["high"][i], data["open"][i])
            self.assertGreaterEqual(data["high"][i], data["close"][i])
            self.assertLessEqual(data["low"][i], data["open"][i])
            self.assertLessEqual(data["low"][i], data["close"][i])
            self.assertGreaterEqual(data["high"][i], data["low"][i])
        
        # Verify starting price
        self.assertEqual(data["open"][0], start_price)


class TestPerformanceAndEdgeCases(TestCase):
    """Test performance and edge cases."""
    
    def setUp(self):
        self.calc = TechnicalIndicators()
    
    def test_large_dataset_performance(self):
        """Test performance with large datasets."""
        import time
        
        # Create large dataset (simulating 1 year of 1-minute data)
        large_data = create_sample_data(525600)  # 365 * 24 * 60
        
        start_time = time.time()
        
        # Test multiple indicators
        sma_result = self.calc.sma(large_data["close"], 20)
        ema_result = self.calc.ema(large_data["close"], 20)
        rsi_result = self.calc.rsi(large_data["close"], 14)
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Should complete within reasonable time (< 10 seconds for large dataset)
        self.assertLess(execution_time, 10.0)
        
        # Verify results are valid
        self.assertEqual(len(sma_result.values), 525600)
        self.assertEqual(len(ema_result.values), 525600)
        self.assertEqual(len(rsi_result.values), 525600)
    
    def test_edge_cases(self):
        """Test edge cases and error conditions."""
        # Test with constant prices (no volatility)
        constant_prices = np.full(50, 1.0000)
        
        rsi_result = self.calc.rsi(constant_prices, 14)
        
        # RSI should handle constant prices gracefully
        valid_values = rsi_result.values[~np.isnan(rsi_result.values)]
        if len(valid_values) > 0:
            # All values should be around 50 (neutral) for constant prices
            self.assertTrue(np.all(np.abs(valid_values - 50.0) < 1.0))
        
        # Test with very small dataset
        small_data = np.array([1.0, 1.1])
        
        with self.assertRaises(ValueError):
            self.calc.sma(small_data, 5)  # Period longer than data
    
    def test_nan_handling(self):
        """Test handling of NaN values in input data."""
        data_with_nan = np.array([1.0, 2.0, np.nan, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        
        # Should warn about NaN values but still process
        with self.assertWarns(UserWarning):
            sma_result = self.calc.sma(data_with_nan, 5)
        
        # Result should be computed
        self.assertEqual(len(sma_result.values), len(data_with_nan))


def run_comprehensive_test():
    """Run comprehensive test of all indicators."""
    print("Running Technical Indicators Comprehensive Test")
    print("=" * 60)
    
    calc = TechnicalIndicators()
    sample_data = create_sample_data(200)
    
    # Test all indicators
    indicators = {}
    
    print("Testing indicators...")
    
    # Basic indicators
    indicators["SMA_20"] = calc.sma(sample_data["close"], 20)
    indicators["EMA_12"] = calc.ema(sample_data["close"], 12)
    indicators["RSI_14"] = calc.rsi(sample_data["close"], 14)
    indicators["MACD"] = calc.macd(sample_data["close"])
    indicators["BB_20"] = calc.bollinger_bands(sample_data["close"], 20, 2.0)
    
    # OHLC-based indicators
    indicators["ATR_14"] = calc.atr(
        sample_data["high"], sample_data["low"], sample_data["close"], 14
    )
    indicators["Stoch"] = calc.stochastic(
        sample_data["high"], sample_data["low"], sample_data["close"], 14, 3
    )
    indicators["Williams_R"] = calc.williams_r(
        sample_data["high"], sample_data["low"], sample_data["close"], 14
    )
    indicators["CCI_20"] = calc.cci(
        sample_data["high"], sample_data["low"], sample_data["close"], 20
    )
    
    # Generate validation report
    validator = IndicatorValidator()
    report = validator.generate_validation_report(indicators)
    
    print(f"\nValidation Report:")
    print(f"Total indicators tested: {report['total_indicators']}")
    print(f"Validation timestamp: {report['validation_timestamp']}")
    
    print("\nIndicator Details:")
    for name, details in report["indicators"].items():
        print(f"  {name}:")
        print(f"    Parameters: {details['parameters']}")
        print(f"    Completeness: {details['completeness_percent']:.1f}%")
        print(f"    Valid points: {details['valid_points']}/{details['total_points']}")
    
    print("\n" + "=" * 60)
    print("All indicators tested successfully!")
    
    return report


if __name__ == "__main__":
    # Run comprehensive test when called directly
    run_comprehensive_test()
    
    # Run unit tests
    import unittest
    unittest.main(argv=[''], exit=False, verbosity=2)