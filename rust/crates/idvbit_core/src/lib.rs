//! IDVBit Core Library
//! 
//! This crate implements the fundamental IDVBit (Infinite-Dimensional Vector Bit) 
//! representations and operations for the ISVGPU project.
//! 
//! ## Mathematical Foundations
//! 
//! IDVBits are modeled as formal objects G(x) = Σ_{n≥0} a_n x^n representing
//! countable families of basis states {φ_n}.
//! 
//! ## Status: SPECULATIVE/HEURISTIC
//! 
//! The concept of true infinite-dimensional computation contradicts proven
//! computational theory. This implementation provides practical approximations
//! and finite representations that emulate some desired properties.

pub mod generating_functions;
pub mod decision_diagrams; 
pub mod formal_series;
pub mod idvbit;

// Re-exports for convenience
pub use idvbit::{IDVBit, IDVBitStorage};
pub use formal_series::{FormalPowerSeries, RationalGeneratingFunction};

/// Core error types for IDVBit operations
#[derive(Debug, thiserror::Error)]
pub enum IDVBitError {
    #[error("Mathematical error: {0}")]
    Mathematical(String),
    
    #[error("Representation error: {0}")]
    Representation(String),
    
    #[error("Serialization error: {0}")]  
    Serialization(String),
    
    #[error("Resource limit exceeded: {0}")]
    ResourceLimit(String),
}

/// Result type for IDVBit operations
pub type Result<T> = std::result::Result<T, IDVBitError>;

#[cfg(test)]
mod tests {
    #[test]
    fn test_basic_functionality() {
        // Basic smoke test - will be expanded in PR-001
        assert_eq!(2 + 2, 4);
    }
}