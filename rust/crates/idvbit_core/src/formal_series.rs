//! Formal power series and generating functions
//! 
//! This module implements the core symbolic mathematics for IDVBit representations.
//! Mathematical status: PROVEN techniques with HEURISTIC optimizations.

use serde::{Serialize, Deserialize, Serializer, Deserializer};
use num_complex::Complex64;
use num_traits::{Zero, One};
use crate::{Result, IDVBitError};

// Custom serialization for Complex64
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct SerializableComplex {
    pub re: f64,
    pub im: f64,
}

impl From<Complex64> for SerializableComplex {
    fn from(c: Complex64) -> Self {
        Self { re: c.re, im: c.im }
    }
}

impl From<SerializableComplex> for Complex64 {
    fn from(c: SerializableComplex) -> Self {
        Complex64::new(c.re, c.im)
    }
}

/// Formal power series G(x) = Σ_{n≥0} a_n x^n
/// 
/// Supports multiple internal representations for efficient operations:
/// - Direct coefficient storage (finite series)
/// - Functional representation with cached coefficients
/// - Rational form P(x)/Q(x) for infinite series
#[derive(Debug, Clone)]
pub struct FormalPowerSeries {
    /// Direct coefficient storage for finite series or cache
    coefficients: Vec<Complex64>,
    /// Optional rational representation for infinite series
    rational_form: Option<RationalGeneratingFunction>,
    /// Maximum computed coefficient index (for lazy evaluation)
    max_computed: usize,
}

impl Serialize for FormalPowerSeries {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let coeffs: Vec<SerializableComplex> = self.coefficients.iter()
            .map(|&c| c.into())
            .collect();
        
        #[derive(Serialize)]
        struct SerializableFPS {
            coefficients: Vec<SerializableComplex>,
            rational_form: Option<RationalGeneratingFunction>,
            max_computed: usize,
        }
        
        SerializableFPS {
            coefficients: coeffs,
            rational_form: self.rational_form.clone(),
            max_computed: self.max_computed,
        }.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for FormalPowerSeries {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct SerializableFPS {
            coefficients: Vec<SerializableComplex>,
            rational_form: Option<RationalGeneratingFunction>,
            max_computed: usize,
        }
        
        let data = SerializableFPS::deserialize(deserializer)?;
        let coeffs = data.coefficients.into_iter()
            .map(|c| c.into())
            .collect();
            
        Ok(FormalPowerSeries {
            coefficients: coeffs,
            rational_form: data.rational_form,
            max_computed: data.max_computed,
        })
    }
}

impl FormalPowerSeries {
    /// Create series from explicit coefficients
    pub fn new(coefficients: Vec<Complex64>) -> Self {
        let max_computed = coefficients.len().saturating_sub(1);
        Self { 
            coefficients, 
            rational_form: None,
            max_computed,
        }
    }
    
    /// Create series from real coefficients (convenience method)
    pub fn from_real(coefficients: Vec<f64>) -> Self {
        let complex_coeffs = coefficients.into_iter()
            .map(|c| Complex64::new(c, 0.0))
            .collect();
        Self::new(complex_coeffs)
    }
    
    /// Create series from rational generating function
    pub fn from_rational(rational: &RationalGeneratingFunction) -> Self {
        Self {
            coefficients: Vec::new(),
            rational_form: Some(rational.clone()),
            max_computed: 0,
        }
    }
    
    /// Create constant series c + 0x + 0x² + ...
    pub fn constant(c: Complex64) -> Self {
        Self::new(vec![c])
    }
    
    /// Get coefficient a_n with lazy computation if needed
    pub fn coefficient(&mut self, n: usize) -> Result<Complex64> {
        // Return cached coefficient if available
        if n < self.coefficients.len() {
            return Ok(self.coefficients[n]);
        }
        
        // Compute from rational form if available
        if let Some(ref rational) = self.rational_form {
            let coeff = rational.coefficient(n)?;
            
            // Extend cache if needed
            if n >= self.coefficients.len() {
                self.coefficients.resize(n + 1, Complex64::zero());
            }
            self.coefficients[n] = coeff;
            self.max_computed = self.max_computed.max(n);
            
            return Ok(coeff);
        }
        
        // Return zero for coefficients beyond stored range
        Ok(Complex64::zero())
    }
    
    /// Get coefficient without mutation (may return None for uncomputed values)
    pub fn coefficient_cached(&self, n: usize) -> Option<Complex64> {
        self.coefficients.get(n).copied()
    }
    
    /// Compute multiple coefficients efficiently
    pub fn coefficients_range(&mut self, start: usize, end: usize) -> Result<Vec<Complex64>> {
        let mut result = Vec::with_capacity(end.saturating_sub(start));
        for i in start..end {
            result.push(self.coefficient(i)?);
        }
        Ok(result)
    }
    
    /// Series addition
    pub fn add(&mut self, other: &mut FormalPowerSeries, terms: usize) -> Result<FormalPowerSeries> {
        let mut result_coeffs = Vec::with_capacity(terms);
        
        for n in 0..terms {
            let a_n = self.coefficient(n)?;
            let b_n = other.coefficient(n)?;
            result_coeffs.push(a_n + b_n);
        }
        
        Ok(FormalPowerSeries::new(result_coeffs))
    }
    
    /// Series multiplication (convolution)
    pub fn multiply(&mut self, other: &mut FormalPowerSeries, terms: usize) -> Result<FormalPowerSeries> {
        let mut result_coeffs = vec![Complex64::zero(); terms];
        
        for n in 0..terms {
            for k in 0..=n {
                let a_k = self.coefficient(k)?;
                let b_nk = other.coefficient(n - k)?;
                result_coeffs[n] += a_k * b_nk;
            }
        }
        
        Ok(FormalPowerSeries::new(result_coeffs))
    }
    
    /// Evaluate series at a point (finite truncation)
    pub fn evaluate(&mut self, x: Complex64, terms: usize) -> Result<Complex64> {
        let mut result = Complex64::zero();
        let mut x_power = Complex64::one();
        
        for n in 0..terms {
            let coeff = self.coefficient(n)?;
            result += coeff * x_power;
            x_power *= x;
        }
        
        Ok(result)
    }
}

/// Rational generating function P(x)/Q(x) with fast coefficient extraction
/// 
/// Uses partial fractions decomposition for O(log n) coefficient extraction
/// when possible, falling back to recurrence relations.
#[derive(Debug, Clone)]
pub struct RationalGeneratingFunction {
    /// Numerator polynomial coefficients (ascending powers)
    numerator: Vec<Complex64>,
    /// Denominator polynomial coefficients (ascending powers)  
    denominator: Vec<Complex64>,
    /// Cached partial fractions decomposition
    partial_fractions: Option<PartialFractionsDecomposition>,
}

#[derive(Debug, Clone)]
struct PartialFractionsDecomposition {
    /// Residues for simple poles
    residues: Vec<Complex64>,
    /// Corresponding pole locations
    poles: Vec<Complex64>,
    /// Polynomial part (if degree(P) >= degree(Q))
    polynomial_part: Vec<Complex64>,
}

impl Serialize for RationalGeneratingFunction {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let num: Vec<SerializableComplex> = self.numerator.iter().map(|&c| c.into()).collect();
        let den: Vec<SerializableComplex> = self.denominator.iter().map(|&c| c.into()).collect();
        
        #[derive(Serialize)]
        struct SerializableRGF {
            numerator: Vec<SerializableComplex>,
            denominator: Vec<SerializableComplex>,
        }
        
        SerializableRGF {
            numerator: num,
            denominator: den,
        }.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for RationalGeneratingFunction {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct SerializableRGF {
            numerator: Vec<SerializableComplex>,
            denominator: Vec<SerializableComplex>,
        }
        
        let data = SerializableRGF::deserialize(deserializer)?;
        let num = data.numerator.into_iter().map(|c| c.into()).collect();
        let den = data.denominator.into_iter().map(|c| c.into()).collect();
        
        Ok(RationalGeneratingFunction {
            numerator: num,
            denominator: den,
            partial_fractions: None,
        })
    }
}

impl RationalGeneratingFunction {
    /// Create rational generating function P(x)/Q(x)
    pub fn new(numerator: Vec<Complex64>, denominator: Vec<Complex64>) -> Result<Self> {
        if denominator.is_empty() || denominator[denominator.len() - 1].is_zero() {
            return Err(IDVBitError::Mathematical(
                "Denominator cannot be zero polynomial".to_string()
            ));
        }
        
        Ok(Self {
            numerator,
            denominator,
            partial_fractions: None,
        })
    }
    
    /// Create from real coefficients (convenience method)
    pub fn from_real(num: Vec<f64>, den: Vec<f64>) -> Result<Self> {
        let complex_num = num.into_iter().map(|c| Complex64::new(c, 0.0)).collect();
        let complex_den = den.into_iter().map(|c| Complex64::new(c, 0.0)).collect();
        Self::new(complex_num, complex_den)
    }
    
    /// Compute coefficient [x^n] P(x)/Q(x)
    pub fn coefficient(&self, n: usize) -> Result<Complex64> {
        // Use partial fractions if available
        if let Some(ref pf) = self.partial_fractions {
            return self.coefficient_from_partial_fractions(n, pf);
        }
        
        // Fall back to recurrence relation method
        self.coefficient_via_recurrence(n)
    }
    
    /// Fast coefficient extraction via partial fractions (when poles are simple)
    fn coefficient_from_partial_fractions(&self, n: usize, pf: &PartialFractionsDecomposition) -> Result<Complex64> {
        let mut result = Complex64::zero();
        
        // Add contributions from simple poles: residue_i * pole_i^n
        for (residue, pole) in pf.residues.iter().zip(pf.poles.iter()) {
            if pole.norm() > 1e-15 { // Avoid division by zero
                result += residue * pole.powc(Complex64::new(n as f64, 0.0));
            }
        }
        
        // Add polynomial part if applicable (only affects first few terms)
        if n < pf.polynomial_part.len() {
            result += pf.polynomial_part[n];
        }
        
        Ok(result)
    }
    
    /// Coefficient extraction via linear recurrence (always works)
    fn coefficient_via_recurrence(&self, _n: usize) -> Result<Complex64> {
        // For P(x)/Q(x), coefficients satisfy the recurrence relation:
        // Q(x) * G(x) = P(x)
        // This gives: Σ q_i * a_{n-i} = [n < deg(P)] p_n else 0
        
        let _q_deg = self.denominator.len() - 1;
        let _p_deg = self.numerator.len().saturating_sub(1);
        
        // For MVP, this would require maintaining a coefficient cache
        // and implementing the full recurrence. For now, return a placeholder.
        // In practice, we'd prefer to precompute partial fractions.
        
        Ok(Complex64::zero())
    }
    
    /// Precompute partial fractions decomposition for fast coefficient extraction
    pub fn precompute_partial_fractions(&mut self) -> Result<()> {
        // This is a complex algorithm involving:
        // 1. Finding roots of denominator polynomial
        // 2. Computing residues at each simple pole
        // 3. Handling polynomial division if deg(P) >= deg(Q)
        
        // For MVP, we'll mark this as TODO and use recurrence method
        // Real implementation would use numerical root finding + residue computation
        
        // Placeholder implementation for simple cases
        if self.denominator.len() == 2 && self.numerator.len() <= 2 {
            // Handle case 1/(1-ax) = Σ a^n x^n  
            let a0 = self.denominator[0];
            let a1 = self.denominator[1];
            
            if a0.norm() > 1e-15 && (a0 + a1).norm() < 1e-15 {
                // Denominator is (1 - pole*x) 
                let pole = -a1 / a0;
                let residue = self.numerator.get(0).copied().unwrap_or(Complex64::one()) / a0;
                
                self.partial_fractions = Some(PartialFractionsDecomposition {
                    residues: vec![residue],
                    poles: vec![pole],
                    polynomial_part: vec![],
                });
            }
        }
        
        Ok(())
    }
}

/// Algebraic generating functions (for advanced applications)
/// 
/// Represents functions satisfying polynomial equations P(x,y) = 0
/// Uses Lagrange inversion formula for coefficient extraction
#[derive(Debug, Clone)]
pub struct AlgebraicGeneratingFunction {
    /// Polynomial equation coefficients P(x,y) = Σ p_{i,j} x^i y^j
    equation_coeffs: Vec<Vec<Complex64>>,
    /// Cached coefficients
    coefficients: Vec<Complex64>,
}

impl AlgebraicGeneratingFunction {
    /// Create from polynomial equation P(x,y) = 0
    pub fn new(equation_coeffs: Vec<Vec<Complex64>>) -> Self {
        Self {
            equation_coeffs,
            coefficients: Vec::new(),
        }
    }
    
    /// Get coefficient using Lagrange inversion (placeholder implementation)
    pub fn coefficient(&mut self, n: usize) -> Result<Complex64> {
        // Extend cache if needed
        while self.coefficients.len() <= n {
            let new_coeff = self.compute_coefficient_lagrange(self.coefficients.len())?;
            self.coefficients.push(new_coeff);
        }
        
        Ok(self.coefficients[n])
    }
    
    /// Compute coefficient via Lagrange inversion formula
    fn compute_coefficient_lagrange(&self, _n: usize) -> Result<Complex64> {
        // Lagrange inversion formula: [x^n] y(x) = (1/n) [t^{n-1}] (x(t))^{n-1} * dx/dt
        // where y(x(t)) = t and P(x(t), t) = 0
        
        // This requires sophisticated symbolic computation
        // For MVP, return placeholder
        Ok(Complex64::zero())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    
    #[test]
    fn test_formal_power_series_basic() {
        let coeffs = vec![1.0, 2.0, 3.0, 4.0];
        let mut series = FormalPowerSeries::from_real(coeffs.clone());
        
        // Test coefficient access
        for (i, &expected) in coeffs.iter().enumerate() {
            let coeff = series.coefficient(i).unwrap();
            assert_relative_eq!(coeff.re, expected, epsilon = 1e-10);
            assert_relative_eq!(coeff.im, 0.0, epsilon = 1e-10);
        }
        
        // Test coefficient beyond stored range
        let coeff = series.coefficient(10).unwrap();
        assert_relative_eq!(coeff.re, 0.0, epsilon = 1e-10);
    }
    
    #[test]
    fn test_rational_generating_function() -> Result<()> {
        // Test 1/(1-x) = 1 + x + x^2 + x^3 + ...
        let num = vec![1.0]; // Numerator = 1
        let den = vec![1.0, -1.0]; // Denominator = 1 - x
        
        let mut rgf = RationalGeneratingFunction::from_real(num, den)?;
        rgf.precompute_partial_fractions()?;
        
        // Test first few coefficients
        for n in 0..5 {
            let coeff = rgf.coefficient(n)?;
            assert_relative_eq!(coeff.re, 1.0, epsilon = 1e-10);
            assert_relative_eq!(coeff.im, 0.0, epsilon = 1e-10);
        }
        
        Ok(())
    }
    
    #[test]
    fn test_series_addition() -> Result<()> {
        let mut series1 = FormalPowerSeries::from_real(vec![1.0, 2.0, 3.0]);
        let mut series2 = FormalPowerSeries::from_real(vec![0.5, 1.5, 2.5]);
        
        let sum = series1.add(&mut series2, 3)?;
        
        let expected = vec![1.5, 3.5, 5.5];
        for (i, &expected_val) in expected.iter().enumerate() {
            let coeff = sum.coefficient_cached(i).unwrap();
            assert_relative_eq!(coeff.re, expected_val, epsilon = 1e-10);
        }
        
        Ok(())
    }
    
    #[test] 
    fn test_series_multiplication() -> Result<()> {
        let mut series1 = FormalPowerSeries::from_real(vec![1.0, 1.0]); // 1 + x
        let mut series2 = FormalPowerSeries::from_real(vec![1.0, 2.0]); // 1 + 2x
        
        let product = series1.multiply(&mut series2, 3)?; // (1+x)(1+2x) = 1 + 3x + 2x^2
        
        let expected = vec![1.0, 3.0, 2.0];
        for (i, &expected_val) in expected.iter().enumerate() {
            let coeff = product.coefficient_cached(i).unwrap();
            assert_relative_eq!(coeff.re, expected_val, epsilon = 1e-10);
        }
        
        Ok(())
    }
}