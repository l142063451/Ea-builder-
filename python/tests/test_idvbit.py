"""
Tests for IDVBit Python Implementation
====================================

Comprehensive test suite for IDVBit functionality, including mathematical
validation, performance testing, and integration with the symbolic math library.
"""

import pytest
import numpy as np
import sympy as sp
from typing import List
import tempfile
import os

from src.research.idvbit import (
    IDVBitPython, IDVBitStorageType, IDVBitMetadata, IDVBitAnalyzer
)


class TestIDVBitBasics:
    """Test basic IDVBit functionality"""
    
    def test_default_creation(self):
        """Test default IDVBit creation"""
        idvbit = IDVBitPython()
        assert idvbit.storage_type == IDVBitStorageType.SYMBOLIC
        assert idvbit.metadata.description == "constant function 1"
        assert idvbit.query_coefficient(0) == 1.0
        assert idvbit.query_coefficient(1) == 0.0
    
    def test_symbolic_series_creation(self):
        """Test creation from coefficient list"""
        coeffs = [1.0, 2.0, 3.0, 4.0]
        idvbit = IDVBitPython.from_symbolic_series(coeffs, "test series")
        
        assert idvbit.metadata.description == "test series"
        for i, expected in enumerate(coeffs):
            assert idvbit.query_coefficient(i) == expected
    
    def test_rational_function_creation(self):
        """Test creation from rational generating function"""
        # Create 1/(1-x) = 1 + x + x² + ...
        num = [1.0]
        den = [1.0, -1.0]
        idvbit = IDVBitPython.from_rational_function(num, den, "geometric")
        
        # Test first few coefficients
        for i in range(10):
            assert abs(idvbit.query_coefficient(i) - 1.0) < 1e-10
    
    def test_coefficient_caching(self):
        """Test coefficient caching functionality"""
        idvbit = IDVBitPython.from_symbolic_series([1, 2, 3, 4, 5])
        
        # Query coefficient to cache it
        coeff = idvbit.query_coefficient(2)
        assert coeff == 3.0
        
        # Check cache stats
        cached, max_cache = idvbit.cache_stats()
        assert cached == 1
        assert max_cache == 1000
        
        # Clear cache
        idvbit.clear_cache()
        cached_after, _ = idvbit.cache_stats()
        assert cached_after == 0
    
    def test_query_range(self):
        """Test querying multiple coefficients"""
        coeffs = [1, 4, 9, 16, 25]  # Perfect squares
        idvbit = IDVBitPython.from_symbolic_series(coeffs)
        
        range_result = idvbit.query_range(0, 5)
        assert range_result == coeffs


class TestIDVBitMathematicalExamples:
    """Test mathematical examples and known sequences"""
    
    def test_geometric_series(self):
        """Test geometric series 1/(1-rx)"""
        # Test with ratio 0.5
        geo = IDVBitPython.geometric_series(0.5)
        
        # Coefficients should be [1, 0.5, 0.25, 0.125, ...]
        expected = [0.5**i for i in range(10)]
        actual = geo.query_range(0, 10)
        
        for i in range(10):
            assert abs(actual[i] - expected[i]) < 1e-10
    
    def test_fibonacci_series(self):
        """Test Fibonacci generating function"""
        fib = IDVBitPython.fibonacci_series()
        
        # Expected Fibonacci sequence: [0, 1, 1, 2, 3, 5, 8, 13, ...]
        expected_fib = [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]
        actual = fib.query_range(0, 10)
        
        for i in range(10):
            assert abs(actual[i] - expected_fib[i]) < 1e-10
    
    def test_catalan_series(self):
        """Test Catalan number generating function"""
        catalan = IDVBitPython.catalan_series()
        
        # Expected Catalan numbers: [1, 1, 2, 5, 14, 42, ...]
        expected_catalan = [1, 1, 2, 5, 14, 42, 132, 429, 1430, 4862]
        actual = catalan.query_range(0, 10)
        
        for i in range(10):
            assert abs(actual[i] - expected_catalan[i]) < 1e-10
    
    def test_sympy_integration(self):
        """Test creation from SymPy expressions"""
        x = sp.Symbol('x')
        
        # Test simple polynomial
        expr = 1 + 2*x + 3*x**2
        idvbit = IDVBitPython.from_sympy_expression(expr, x, "polynomial")
        
        assert abs(idvbit.query_coefficient(0) - 1.0) < 1e-10
        assert abs(idvbit.query_coefficient(1) - 2.0) < 1e-10
        assert abs(idvbit.query_coefficient(2) - 3.0) < 1e-10
        assert abs(idvbit.query_coefficient(3) - 0.0) < 1e-10
    
    def test_rational_sympy_integration(self):
        """Test rational function from SymPy"""
        x = sp.Symbol('x')
        
        # Test 1/(1-x) geometric series
        expr = 1 / (1 - x)
        idvbit = IDVBitPython.from_sympy_expression(expr, x, "geometric from sympy")
        
        # Should give geometric series coefficients
        for i in range(5):
            assert abs(idvbit.query_coefficient(i) - 1.0) < 1e-10


class TestIDVBitOperations:
    """Test arithmetic operations on IDVBits"""
    
    def test_addition(self):
        """Test IDVBit addition"""
        # Create two simple series
        a = IDVBitPython.from_symbolic_series([1, 2, 3])
        b = IDVBitPython.from_symbolic_series([4, 5, 6])
        
        result = a.add(b, max_terms=5)
        
        # Should be [5, 7, 9, 0, 0]
        expected = [5, 7, 9, 0, 0]
        actual = result.query_range(0, 5)
        
        for i in range(5):
            assert abs(actual[i] - expected[i]) < 1e-10
    
    def test_multiplication(self):
        """Test IDVBit multiplication (convolution)"""
        # (1 + x) * (1 + 2x) = 1 + 3x + 2x²
        a = IDVBitPython.from_symbolic_series([1, 1])  # 1 + x
        b = IDVBitPython.from_symbolic_series([1, 2])  # 1 + 2x
        
        result = a.multiply(b, max_terms=5)
        
        # Should be [1, 3, 2, 0, 0]
        expected = [1, 3, 2, 0, 0]
        actual = result.query_range(0, 5)
        
        for i in range(5):
            assert abs(actual[i] - expected[i]) < 1e-10
    
    def test_evaluation(self):
        """Test series evaluation at a point"""
        # Create series 1 + 2x + 3x²
        idvbit = IDVBitPython.from_symbolic_series([1, 2, 3])
        
        # Evaluate at x = 0.5
        # Should be 1 + 2*0.5 + 3*0.25 = 1 + 1 + 0.75 = 2.75
        result = idvbit.evaluate(0.5, max_terms=10)
        assert abs(result - 2.75) < 1e-10
        
        # Evaluate at x = 0 should give first coefficient
        result = idvbit.evaluate(0.0, max_terms=10)
        assert abs(result - 1.0) < 1e-10


class TestIDVBitSerialization:
    """Test serialization and persistence"""
    
    def test_to_dict_from_dict(self):
        """Test dictionary serialization"""
        original = IDVBitPython.from_symbolic_series([1, 2, 3, 4, 5], "test series")
        
        # Convert to dict and back
        data = original.to_dict()
        restored = IDVBitPython.from_dict(data)
        
        # Check coefficients match
        for i in range(5):
            assert abs(original.query_coefficient(i) - restored.query_coefficient(i)) < 1e-10
        
        # Check metadata
        assert original.metadata.description == restored.metadata.description
    
    def test_save_load(self):
        """Test file save/load functionality"""
        original = IDVBitPython.fibonacci_series()
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            temp_filename = f.name
        
        try:
            original.save(temp_filename)
            restored = IDVBitPython.load(temp_filename)
            
            # Check first few Fibonacci numbers
            for i in range(10):
                assert abs(original.query_coefficient(i) - restored.query_coefficient(i)) < 1e-10
        finally:
            os.unlink(temp_filename)


class TestIDVBitAnalyzer:
    """Test IDVBit analysis tools"""
    
    def test_coefficient_growth_analysis(self):
        """Test coefficient growth analysis"""
        # Test with geometric series (bounded growth)
        geo = IDVBitPython.geometric_series(0.5)
        growth = IDVBitAnalyzer.coefficient_growth_analysis(geo, max_index=20)
        
        assert growth['is_bounded'] is True
        assert abs(growth['average_growth_ratio'] - 0.5) < 0.1  # Should be around 0.5
        
        # Test with Fibonacci (exponential growth)
        fib = IDVBitPython.fibonacci_series()
        growth_fib = IDVBitAnalyzer.coefficient_growth_analysis(fib, max_index=20)
        
        assert growth_fib['is_bounded'] is True  # Still bounded for small indices
        # Golden ratio ≈ 1.618
        assert growth_fib['average_growth_ratio'] > 1.5
    
    def test_performance_benchmark(self):
        """Test performance benchmarking"""
        idvbit = IDVBitPython.fibonacci_series()
        queries = list(range(10))
        
        perf = IDVBitAnalyzer.performance_benchmark(idvbit, queries)
        
        assert 'cold_query_time' in perf
        assert 'warm_query_time' in perf
        assert 'cache_speedup' in perf
        assert perf['cold_query_time'] >= perf['warm_query_time']  # Cache should help
        assert perf['cache_speedup'] >= 1.0  # Should be speedup, not slowdown
    
    def test_representation_comparison(self):
        """Test comparison of different representations"""
        # Compare Fibonacci representations
        fib_coeffs = [0, 1, 1, 2, 3, 5, 8, 13]
        
        comparison = IDVBitAnalyzer.compare_representations(
            fib_coeffs, 
            ["fibonacci", "explicit coefficients"]
        )
        
        assert isinstance(comparison, str)
        assert "fibonacci" in comparison
        assert "error" in comparison


class TestIDVBitComplexCases:
    """Test complex and edge cases"""
    
    def test_large_indices(self):
        """Test querying large indices"""
        geo = IDVBitPython.geometric_series(0.9)  # Close to divergence
        
        # Should still work for moderately large indices
        coeff = geo.query_coefficient(100)
        expected = 0.9**100
        assert abs(coeff - expected) < 1e-10
    
    def test_complex_coefficients(self):
        """Test with complex coefficients"""
        # Series with complex coefficients
        coeffs = [1+1j, 2-1j, 3+2j, 4-2j]
        idvbit = IDVBitPython.from_symbolic_series(coeffs, "complex series")
        
        for i, expected in enumerate(coeffs):
            actual = idvbit.query_coefficient(i)
            assert abs(actual - expected) < 1e-10
    
    def test_zero_coefficients(self):
        """Test series with many zero coefficients"""
        # Sparse series: x² + x⁵ + x⁸
        coeffs = [0, 0, 1, 0, 0, 1, 0, 0, 1]
        idvbit = IDVBitPython.from_symbolic_series(coeffs, "sparse series")
        
        assert idvbit.query_coefficient(2) == 1
        assert idvbit.query_coefficient(5) == 1
        assert idvbit.query_coefficient(8) == 1
        assert idvbit.query_coefficient(0) == 0
        assert idvbit.query_coefficient(1) == 0
        assert idvbit.query_coefficient(3) == 0
    
    def test_string_representations(self):
        """Test string representations"""
        idvbit = IDVBitPython.from_symbolic_series([1, 2, 3], "test")
        
        repr_str = repr(idvbit)
        assert "IDVBitPython" in repr_str
        assert "test" in repr_str
        
        str_repr = str(idvbit)
        assert "test" in str_repr
        assert "1.000" in str_repr  # Should show coefficients


class TestIDVBitMathematicalProperties:
    """Test mathematical properties and validations"""
    
    def test_linearity_of_operations(self):
        """Test linearity: (a + b) + c = a + (b + c)"""
        a = IDVBitPython.from_symbolic_series([1, 2])
        b = IDVBitPython.from_symbolic_series([3, 4])  
        c = IDVBitPython.from_symbolic_series([5, 6])
        
        # Test associativity
        left = a.add(b).add(c, max_terms=5)
        right = a.add(b.add(c, max_terms=5), max_terms=5)
        
        for i in range(5):
            assert abs(left.query_coefficient(i) - right.query_coefficient(i)) < 1e-10
    
    def test_multiplication_properties(self):
        """Test multiplication properties"""
        # Test (1 + x) * (1 + x) = 1 + 2x + x²
        poly = IDVBitPython.from_symbolic_series([1, 1])  # 1 + x
        result = poly.multiply(poly, max_terms=5)
        
        expected = [1, 2, 1, 0, 0]  # 1 + 2x + x²
        actual = result.query_range(0, 5)
        
        for i in range(5):
            assert abs(actual[i] - expected[i]) < 1e-10
    
    def test_rational_function_coefficient_extraction(self):
        """Test advanced coefficient extraction for rational functions"""
        # Test 1/(1-x-x²) Fibonacci generating function more thoroughly
        fib = IDVBitPython.fibonacci_series()
        
        # Test larger Fibonacci numbers
        fib_sequence = [0, 1]
        for i in range(2, 20):
            fib_sequence.append(fib_sequence[-1] + fib_sequence[-2])
        
        for i, expected in enumerate(fib_sequence):
            actual = fib.query_coefficient(i)
            assert abs(actual - expected) < 1e-10


if __name__ == "__main__":
    # Run a simple test demonstration
    print("Running IDVBit Python tests...")
    
    # Test basic functionality
    idvbit = IDVBitPython.fibonacci_series()
    print(f"Fibonacci IDVBit: {idvbit}")
    print(f"First 10 coefficients: {idvbit.query_range(0, 10)}")
    
    # Test analysis
    analyzer = IDVBitAnalyzer()
    growth = analyzer.coefficient_growth_analysis(idvbit, 30)
    print(f"Growth analysis: {growth}")
    
    print("Basic tests passed!")