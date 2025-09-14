"""
Symbolic mathematics library for ISVGPU research.

This module provides Python interfaces for symbolic computation, formal power series,
and generating functions. It serves as a bridge between SymPy's symbolic capabilities
and the Rust implementations.

Mathematical status: PROVEN techniques with HEURISTIC optimizations.
"""

import sympy as sp
import numpy as np
from typing import List, Optional, Union, Callable
from sympy import symbols, Symbol, Expr, series, limit, oo, I
from sympy.abc import x, n, k
import warnings


class FormalPowerSeries:
    """
    Python wrapper for formal power series with SymPy integration.
    
    Supports symbolic series manipulation and coefficient extraction
    using SymPy's powerful symbolic computation engine.
    """
    
    def __init__(self, expression: Optional[Expr] = None, coefficients: Optional[List[complex]] = None):
        """
        Create a formal power series.
        
        Args:
            expression: SymPy expression for the generating function
            coefficients: Direct coefficient list for finite series
        """
        if expression is not None and coefficients is not None:
            raise ValueError("Cannot specify both expression and coefficients")
        
        self.expression = expression
        self.coefficients = coefficients or []
        self._cached_coeffs = {}
        
    @classmethod
    def from_expression(cls, expr: Expr) -> 'FormalPowerSeries':
        """Create series from SymPy expression."""
        return cls(expression=expr)
    
    @classmethod
    def from_coefficients(cls, coeffs: List[Union[int, float, complex]]) -> 'FormalPowerSeries':
        """Create series from coefficient list."""
        complex_coeffs = [complex(c) for c in coeffs]
        return cls(coefficients=complex_coeffs)
    
    @classmethod
    def from_rational_function(cls, rational: 'RationalGeneratingFunction') -> 'FormalPowerSeries':
        """Create series from rational generating function."""
        # Create from the rational function's SymPy representation
        return cls(expression=rational.expression)
    
    def coefficient(self, n: int) -> complex:
        """
        Extract coefficient [x^n] from the series.
        
        Uses SymPy's series expansion and coefficient extraction.
        """
        if n < 0:
            return 0.0
            
        # Return direct coefficient if available
        if self.coefficients and n < len(self.coefficients):
            return self.coefficients[n]
        
        # Check cache
        if n in self._cached_coeffs:
            return self._cached_coeffs[n]
            
        # Extract from symbolic expression
        if self.expression is not None:
            try:
                # Use series expansion around x=0
                series_expansion = series(self.expression, x, 0, n + 2)
                coeff = series_expansion.coeff(x, n)
                
                if coeff is None:
                    result = 0.0
                else:
                    # Convert to complex number
                    result = complex(coeff.evalf())
                    
                self._cached_coeffs[n] = result
                return result
                
            except Exception as e:
                warnings.warn(f"Failed to extract coefficient {n}: {e}")
                return 0.0
        
        # Default: return 0 for undefined coefficients
        return 0.0
    
    def coefficients_range(self, start: int, end: int) -> List[complex]:
        """Extract multiple coefficients efficiently."""
        return [self.coefficient(i) for i in range(start, end)]
    
    def add(self, other: 'FormalPowerSeries') -> 'FormalPowerSeries':
        """Add two formal power series."""
        if self.expression is not None and other.expression is not None:
            return FormalPowerSeries.from_expression(self.expression + other.expression)
        
        # Fall back to coefficient arithmetic for mixed cases
        max_len = max(
            len(self.coefficients) if self.coefficients else 0,
            len(other.coefficients) if other.coefficients else 0,
            10  # Default expansion depth
        )
        
        result_coeffs = []
        for i in range(max_len):
            result_coeffs.append(self.coefficient(i) + other.coefficient(i))
        
        return FormalPowerSeries.from_coefficients(result_coeffs)
    
    def multiply(self, other: 'FormalPowerSeries', terms: int = 10) -> 'FormalPowerSeries':
        """Multiply two formal power series (convolution)."""
        if self.expression is not None and other.expression is not None:
            return FormalPowerSeries.from_expression(self.expression * other.expression)
        
        # Coefficient-based convolution
        result_coeffs = [0.0] * terms
        
        for n in range(terms):
            for k in range(n + 1):
                result_coeffs[n] += self.coefficient(k) * other.coefficient(n - k)
        
        return FormalPowerSeries.from_coefficients(result_coeffs)
    
    def evaluate(self, value: complex, terms: int = 10) -> complex:
        """Evaluate series at a specific point (finite truncation)."""
        if self.expression is not None:
            try:
                return complex(self.expression.subs(x, value).evalf())
            except:
                pass
        
        # Horner's method for numerical evaluation
        result = 0.0
        value_power = 1.0
        
        for i in range(terms):
            result += self.coefficient(i) * value_power
            value_power *= value
            
        return result
    
    def __str__(self) -> str:
        if self.expression is not None:
            return f"FormalPowerSeries({self.expression})"
        else:
            coeffs_str = str(self.coefficients[:5])
            if len(self.coefficients) > 5:
                coeffs_str = coeffs_str[:-1] + ", ...]"
            return f"FormalPowerSeries(coefficients={coeffs_str})"


class RationalGeneratingFunction:
    """
    Rational generating function P(x)/Q(x) with fast coefficient extraction.
    
    Uses SymPy's symbolic capabilities for partial fractions decomposition
    and closed-form coefficient extraction when possible.
    """
    
    def __init__(self, numerator: Expr, denominator: Expr):
        """
        Create rational generating function P(x)/Q(x).
        
        Args:
            numerator: SymPy expression for P(x)
            denominator: SymPy expression for Q(x)
        """
        self.numerator = numerator
        self.denominator = denominator
        self.expression = numerator / denominator
        self._partial_fractions = None
        self._cached_coeffs = {}
    
    @classmethod
    def from_coefficients(cls, num_coeffs: List[float], den_coeffs: List[float]) -> 'RationalGeneratingFunction':
        """Create from polynomial coefficient lists."""
        num_expr = sum(c * x**i for i, c in enumerate(num_coeffs) if c != 0)
        den_expr = sum(c * x**i for i, c in enumerate(den_coeffs) if c != 0)
        return cls(num_expr, den_expr)
    
    def precompute_partial_fractions(self) -> bool:
        """
        Precompute partial fractions decomposition for fast coefficient extraction.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            self._partial_fractions = sp.apart(self.expression, x)
            return True
        except Exception:
            return False
    
    def coefficient(self, n: int) -> complex:
        """
        Extract coefficient [x^n] using partial fractions when possible,
        falling back to series expansion.
        """
        if n < 0:
            return 0.0
            
        # Check cache
        if n in self._cached_coeffs:
            return self._cached_coeffs[n]
        
        result = 0.0
        
        try:
            # Use partial fractions if available
            if self._partial_fractions is not None:
                result = self._coefficient_from_partial_fractions(n)
            else:
                # Fall back to series expansion
                series_expansion = series(self.expression, x, 0, n + 2)
                coeff = series_expansion.coeff(x, n)
                result = complex(coeff.evalf()) if coeff is not None else 0.0
                
            self._cached_coeffs[n] = result
            return result
            
        except Exception as e:
            warnings.warn(f"Failed to extract coefficient {n}: {e}")
            return 0.0
    
    def _coefficient_from_partial_fractions(self, n: int) -> complex:
        """Extract coefficient using partial fractions decomposition."""
        # This is a simplified implementation
        # In practice, we'd need to handle the full partial fractions structure
        try:
            series_expansion = series(self._partial_fractions, x, 0, n + 2)
            coeff = series_expansion.coeff(x, n)
            return complex(coeff.evalf()) if coeff is not None else 0.0
        except:
            return 0.0
    
    def to_formal_series(self) -> FormalPowerSeries:
        """Convert to FormalPowerSeries for unified interface."""
        return FormalPowerSeries.from_expression(self.expression)
    
    def __str__(self) -> str:
        return f"RationalGeneratingFunction(({self.numerator})/({self.denominator}))"


class GeneratingFunctionToolkit:
    """
    Collection of utilities for generating function manipulation and analysis.
    """
    
    @staticmethod
    def geometric_series(a: Union[int, float, Expr] = 1) -> RationalGeneratingFunction:
        """
        Create geometric series 1/(1-ax) = Σ (ax)^n.
        
        Args:
            a: Coefficient (default 1 for 1/(1-x))
        """
        return RationalGeneratingFunction(1, 1 - a * x)
    
    @staticmethod
    def binomial_series(alpha: Union[int, float, Expr]) -> FormalPowerSeries:
        """
        Create binomial series (1+x)^α = Σ C(α,n) x^n.
        """
        expr = (1 + x) ** alpha
        return FormalPowerSeries.from_expression(expr)
    
    @staticmethod
    def exponential_generating_function(f: Expr) -> FormalPowerSeries:
        """
        Create exponential generating function e^(f(x)).
        """
        expr = sp.exp(f)
        return FormalPowerSeries.from_expression(expr)
    
    @staticmethod
    def fibonacci_generating_function() -> RationalGeneratingFunction:
        """
        Create Fibonacci generating function x/(1-x-x^2).
        """
        return RationalGeneratingFunction(x, 1 - x - x**2)
    
    @staticmethod
    def catalan_generating_function() -> FormalPowerSeries:
        """
        Create Catalan generating function (1-√(1-4x))/(2x).
        """
        expr = (1 - sp.sqrt(1 - 4*x)) / (2*x)
        return FormalPowerSeries.from_expression(expr)
    
    @staticmethod
    def coefficient_extraction_benchmark(gf: Union[FormalPowerSeries, RationalGeneratingFunction], 
                                       max_n: int = 100) -> dict:
        """
        Benchmark coefficient extraction performance and accuracy.
        
        Returns:
            Dictionary with timing and accuracy statistics
        """
        import time
        
        start_time = time.time()
        coefficients = []
        
        for n in range(max_n + 1):
            coefficients.append(gf.coefficient(n))
            
        end_time = time.time()
        
        return {
            'total_time': end_time - start_time,
            'avg_time_per_coeff': (end_time - start_time) / (max_n + 1),
            'coefficients_computed': len(coefficients),
            'first_few_coeffs': coefficients[:10],
            'last_few_coeffs': coefficients[-5:] if len(coefficients) >= 5 else coefficients
        }


def create_d_finite_series(recurrence_coeffs: List[float], 
                          initial_values: List[complex],
                          name: str = "D-finite series") -> FormalPowerSeries:
    """
    Create a D-finite series from linear recurrence relation.
    
    A D-finite series satisfies: Σ p_i(n) a_{n+i} = 0
    where p_i are polynomials.
    
    Args:
        recurrence_coeffs: Coefficients for the recurrence relation
        initial_values: Initial values a_0, a_1, ..., a_{r-1}
        name: Description for the series
    
    Returns:
        FormalPowerSeries representing the D-finite series
    """
    # This is a placeholder implementation
    # Full implementation would solve the recurrence symbolically
    
    def coefficient_func(n: int) -> complex:
        if n < len(initial_values):
            return initial_values[n]
        
        # Apply recurrence relation (simplified)
        # In practice, we'd solve the characteristic equation
        return 0.0
    
    # Create coefficient list using the recurrence
    coeffs = []
    for i in range(50):  # Compute first 50 terms
        if i < len(initial_values):
            coeffs.append(initial_values[i])
        else:
            # Apply simple recurrence (placeholder)
            val = sum(recurrence_coeffs[j] * coeffs[i-len(recurrence_coeffs)+j] 
                     for j in range(len(recurrence_coeffs)) 
                     if i-len(recurrence_coeffs)+j >= 0)
            coeffs.append(val)
    
    return FormalPowerSeries.from_coefficients(coeffs)


# Example usage and test cases
def run_examples():
    """Run example computations to demonstrate the library."""
    
    print("=== ISVGPU Symbolic Math Library Examples ===\n")
    
    # Example 1: Geometric series
    print("1. Geometric Series 1/(1-x):")
    geometric = GeneratingFunctionToolkit.geometric_series()
    geometric.precompute_partial_fractions()
    
    print(f"First 10 coefficients: {[geometric.coefficient(n) for n in range(10)]}")
    print(f"Expected: {[1.0] * 10}")
    print()
    
    # Example 2: Fibonacci series
    print("2. Fibonacci Generating Function x/(1-x-x^2):")
    fibonacci = GeneratingFunctionToolkit.fibonacci_generating_function()
    fib_coeffs = [fibonacci.coefficient(n) for n in range(10)]
    print(f"First 10 Fibonacci numbers: {fib_coeffs}")
    print()
    
    # Example 3: Series arithmetic
    print("3. Series Arithmetic:")
    series1 = FormalPowerSeries.from_coefficients([1, 1, 1])  # 1 + x + x^2
    series2 = FormalPowerSeries.from_coefficients([1, 2, 3])  # 1 + 2x + 3x^2
    
    sum_series = series1.add(series2)
    product_series = series1.multiply(series2, terms=6)
    
    print(f"Sum coefficients: {sum_series.coefficients}")
    print(f"Product coefficients: {product_series.coefficients_range(0, 6)}")
    print()
    
    # Example 4: Performance benchmark
    print("4. Performance Benchmark:")
    benchmark_results = GeneratingFunctionToolkit.coefficient_extraction_benchmark(geometric, max_n=50)
    print(f"Total time for 51 coefficients: {benchmark_results['total_time']:.4f}s")
    print(f"Average time per coefficient: {benchmark_results['avg_time_per_coeff']:.6f}s")
    print()


if __name__ == "__main__":
    run_examples()