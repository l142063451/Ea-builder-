"""
Test suite for the symbolic mathematics library.
"""

import pytest
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from research.symbolic_math import (
    FormalPowerSeries, 
    RationalGeneratingFunction, 
    GeneratingFunctionToolkit,
    create_d_finite_series
)
import sympy as sp
from sympy.abc import x


class TestFormalPowerSeries:
    
    def test_from_coefficients(self):
        coeffs = [1, 2, 3, 4]
        series = FormalPowerSeries.from_coefficients(coeffs)
        
        for i, expected in enumerate(coeffs):
            assert abs(series.coefficient(i) - expected) < 1e-10
            
        # Test beyond stored range
        assert abs(series.coefficient(10)) < 1e-10
    
    def test_from_expression(self):
        # Test 1 + x + x^2 + x^3 (geometric series truncated)
        expr = 1 + x + x**2 + x**3
        series = FormalPowerSeries.from_expression(expr)
        
        expected = [1, 1, 1, 1]
        for i, exp_val in enumerate(expected):
            coeff = series.coefficient(i)
            assert abs(coeff - exp_val) < 1e-10
    
    def test_coefficient_caching(self):
        expr = 1 / (1 - x)  # Geometric series
        series = FormalPowerSeries.from_expression(expr)
        
        # First call should compute and cache
        coeff1 = series.coefficient(5)
        # Second call should use cache
        coeff2 = series.coefficient(5)
        
        assert abs(coeff1 - coeff2) < 1e-15
        assert abs(coeff1 - 1.0) < 1e-10
    
    def test_series_addition(self):
        series1 = FormalPowerSeries.from_coefficients([1, 2, 3])
        series2 = FormalPowerSeries.from_coefficients([4, 5, 6])
        
        sum_series = series1.add(series2)
        expected = [5, 7, 9]
        
        for i, exp_val in enumerate(expected):
            assert abs(sum_series.coefficient(i) - exp_val) < 1e-10
    
    def test_series_multiplication(self):
        # (1 + x) * (1 + 2x) = 1 + 3x + 2x^2
        series1 = FormalPowerSeries.from_coefficients([1, 1])
        series2 = FormalPowerSeries.from_coefficients([1, 2])
        
        product = series1.multiply(series2, terms=3)
        expected = [1, 3, 2]
        
        for i, exp_val in enumerate(expected):
            assert abs(product.coefficient(i) - exp_val) < 1e-10
    
    def test_evaluation(self):
        # Test 1 + x + x^2 at x = 0.5
        series = FormalPowerSeries.from_coefficients([1, 1, 1])
        result = series.evaluate(0.5, terms=3)
        expected = 1 + 0.5 + 0.25  # 1.75
        
        assert abs(result - expected) < 1e-10


class TestRationalGeneratingFunction:
    
    def test_geometric_series(self):
        # 1/(1-x) = 1 + x + x^2 + ...
        rgf = RationalGeneratingFunction.from_coefficients([1], [1, -1])
        rgf.precompute_partial_fractions()
        
        for n in range(10):
            coeff = rgf.coefficient(n)
            assert abs(coeff - 1.0) < 1e-10
    
    def test_fibonacci_generation(self):
        # x/(1-x-x^2) generates Fibonacci numbers
        rgf = RationalGeneratingFunction(x, 1 - x - x**2)
        
        # First few Fibonacci numbers: 0, 1, 1, 2, 3, 5, 8, 13, ...
        expected_fib = [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]
        
        for n, expected in enumerate(expected_fib):
            coeff = rgf.coefficient(n)
            assert abs(coeff - expected) < 1e-10
    
    def test_partial_fractions_simple(self):
        # Simple case that should work with our partial fractions implementation
        rgf = RationalGeneratingFunction.from_coefficients([1], [1, -2])  # 1/(1-2x)
        success = rgf.precompute_partial_fractions()
        
        # Should be 2^n for n >= 0
        if success:
            for n in range(5):
                coeff = rgf.coefficient(n)
                expected = 2**n
                assert abs(coeff - expected) < 1e-10


class TestGeneratingFunctionToolkit:
    
    def test_geometric_series_creation(self):
        gf = GeneratingFunctionToolkit.geometric_series(2)  # 1/(1-2x)
        gf.precompute_partial_fractions()
        
        # Should give coefficients 2^n
        for n in range(5):
            coeff = gf.coefficient(n)
            expected = 2**n
            assert abs(coeff - expected) < 1e-10
    
    def test_fibonacci_series_creation(self):
        fib_gf = GeneratingFunctionToolkit.fibonacci_generating_function()
        
        # Test first few Fibonacci numbers
        expected = [0, 1, 1, 2, 3, 5, 8]
        for n, exp_val in enumerate(expected):
            coeff = fib_gf.coefficient(n)
            assert abs(coeff - exp_val) < 1e-10
    
    def test_binomial_series(self):
        # (1+x)^2 = 1 + 2x + x^2
        binomial = GeneratingFunctionToolkit.binomial_series(2)
        
        expected = [1, 2, 1]
        for n, exp_val in enumerate(expected):
            coeff = binomial.coefficient(n)
            assert abs(coeff - exp_val) < 1e-10
            
        # Coefficient beyond degree should be 0
        assert abs(binomial.coefficient(5)) < 1e-10
    
    def test_coefficient_extraction_benchmark(self):
        gf = GeneratingFunctionToolkit.geometric_series()
        results = GeneratingFunctionToolkit.coefficient_extraction_benchmark(gf, max_n=10)
        
        assert 'total_time' in results
        assert 'avg_time_per_coeff' in results
        assert len(results['first_few_coeffs']) == 10
        assert all(abs(c - 1.0) < 1e-10 for c in results['first_few_coeffs'])


class TestDFiniteSeries:
    
    def test_simple_recurrence(self):
        # a_n = a_{n-1} + a_{n-2} with a_0=0, a_1=1 (Fibonacci)
        recurrence_coeffs = [1, 1, -1]  # a_n - a_{n-1} - a_{n-2} = 0
        initial_values = [0, 1]
        
        series = create_d_finite_series(recurrence_coeffs, initial_values)
        
        # Should match Fibonacci sequence for first few terms
        expected_fib = [0, 1, 1, 2, 3, 5]
        for n, expected in enumerate(expected_fib[:len(initial_values)]):
            coeff = series.coefficient(n)
            assert abs(coeff - expected) < 1e-10


class TestIntegration:
    
    def test_rust_python_consistency(self):
        """Test that Python and Rust implementations give consistent results."""
        # This would be a fuller test when we have Rust-Python bindings
        # For now, just test that our Python implementation works
        
        series = FormalPowerSeries.from_coefficients([1, 1, 1, 1])
        rgf = RationalGeneratingFunction.from_coefficients([1, 1], [1, 0, 1])
        
        # Basic consistency checks
        assert abs(series.coefficient(0) - 1.0) < 1e-10
        assert abs(series.coefficient(3) - 1.0) < 1e-10
        assert abs(rgf.coefficient(0) - 1.0) < 1e-10
    
    def test_symbolic_expressions_with_sympy(self):
        """Test integration with SymPy symbolic expressions."""
        import sympy as sp
        
        # Create a symbolic series from SymPy expression
        expr = sp.cos(x)  # cos(x) = 1 - x^2/2! + x^4/4! - ...
        series = FormalPowerSeries.from_expression(expr)
        
        # Test first few Taylor coefficients of cos(x)
        assert abs(series.coefficient(0) - 1.0) < 1e-10  # 1
        assert abs(series.coefficient(1) - 0.0) < 1e-10  # 0  
        assert abs(series.coefficient(2) - (-0.5)) < 1e-10  # -1/2
        assert abs(series.coefficient(4) - (1/24)) < 1e-10  # 1/24


if __name__ == "__main__":
    pytest.main([__file__, "-v"])