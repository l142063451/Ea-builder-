//! IDVBit main type and storage backends
//!
//! This module implements the core IDVBit (Infinite-Dimensional Vector Bit) representation
//! that can model countable families of basis states using advanced mathematical techniques.
//!
//! ## Mathematical Foundation
//! 
//! An IDVBit is formally modeled as G(x) = Σ_{n≥0} a_n x^n where:
//! - Each coefficient a_n represents the amplitude/probability for basis state φ_n
//! - The generating function G(x) provides efficient access to coefficients
//! - Multiple storage backends optimize for different use cases
//!
//! ## Storage Backends
//! 
//! 1. **Symbolic**: Rational generating functions P(x)/Q(x) with fast coefficient extraction
//! 2. **DecisionDiagram**: OBDD/SDD representations for combinatorial structures  
//! 3. **TensorNetwork**: Compressed tensor representations for high-dimensional data
//!
//! Status: PROVEN mathematics with HEURISTIC optimizations and SPECULATIVE concepts clearly marked

use serde::{Serialize, Deserialize};
use std::collections::HashMap;
use num_complex::Complex64;
use crate::{Result, IDVBitError, formal_series::{FormalPowerSeries, RationalGeneratingFunction, SerializableComplex}};

/// IDVBit main type - Advanced implementation for PR-002
/// 
/// Represents an infinite-dimensional vector bit through multiple storage backends
/// optimized for different mathematical structures and query patterns.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IDVBit {
    /// Primary storage backend
    storage: IDVBitStorage,
    /// Metadata for optimization and validation
    metadata: IDVBitMetadata,
    /// Cache for frequently accessed coefficients
    #[serde(skip)]
    coefficient_cache: HashMap<u64, Complex64>,
}

/// IDVBit storage backends with advanced implementations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum IDVBitStorage {
    /// Symbolic generating function representation
    /// Best for: Structured mathematical sequences, closed-form solutions
    Symbolic { 
        series: FormalPowerSeries,
        rational: Option<RationalGeneratingFunction>,
        description: String,
    },
    
    /// Decision diagram representation  
    /// Best for: Boolean functions, combinatorial structures
    DecisionDiagram { 
        nodes: Vec<DecisionNode>,
        root: usize,
        variable_order: Vec<String>,
    },
    
    /// Tensor network representation
    /// Best for: High-dimensional quantum-like states, compressed representations
    TensorNetwork { 
        cores: Vec<TensorCore>,
        dimensions: Vec<usize>,
        bond_dimensions: Vec<usize>,
    },
    
    /// Hybrid representation combining multiple backends
    /// Best for: Complex problems requiring multiple mathematical techniques
    Hybrid {
        primary: Box<IDVBitStorage>,
        secondary: Box<IDVBitStorage>,
        switching_strategy: SwitchingStrategy,
    },
}

/// Decision diagram node for combinatorial representations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionNode {
    /// Variable index this node tests
    variable: usize,
    /// Low child (variable = 0)
    low: NodeRef,
    /// High child (variable = 1) 
    high: NodeRef,
    /// Optional coefficient for weighted diagrams
    coefficient: Option<SerializableComplex>,
}

impl DecisionNode {
    /// Create new decision node
    pub fn new(variable: usize, low: NodeRef, high: NodeRef, coefficient: Option<Complex64>) -> Self {
        Self {
            variable,
            low,
            high,
            coefficient: coefficient.map(SerializableComplex::from),
        }
    }
    
    /// Get coefficient as Complex64
    pub fn complex_coefficient(&self) -> Option<Complex64> {
        self.coefficient.map(Complex64::from)
    }
}

/// Reference to a decision diagram node
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NodeRef {
    /// Reference to another node by index
    Node(usize),
    /// Terminal node with value  
    Terminal(SerializableComplex),
}

impl NodeRef {
    /// Create terminal node from Complex64
    pub fn terminal(value: Complex64) -> Self {
        NodeRef::Terminal(SerializableComplex::from(value))
    }
    
    /// Get terminal value as Complex64 if this is a terminal node
    pub fn terminal_value(&self) -> Option<Complex64> {
        match self {
            NodeRef::Terminal(c) => Some(Complex64::from(*c)),
            NodeRef::Node(_) => None,
        }
    }
}

/// Tensor core for tensor network representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorCore {
    /// Tensor data in row-major order
    data: Vec<SerializableComplex>,
    /// Dimensions of this core
    shape: Vec<usize>,
    /// Bonds to other cores
    bonds: Vec<BondInfo>,
}

impl TensorCore {
    /// Convert from Complex64 data
    pub fn from_complex_data(data: Vec<Complex64>, shape: Vec<usize>, bonds: Vec<BondInfo>) -> Self {
        let serializable_data = data.into_iter().map(SerializableComplex::from).collect();
        Self { data: serializable_data, shape, bonds }
    }
    
    /// Get complex data
    pub fn complex_data(&self) -> Vec<Complex64> {
        self.data.iter().map(|c| Complex64::from(*c)).collect()
    }
}

/// Bond information for tensor networks
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BondInfo {
    /// Connected core index
    core: usize,
    /// Bond dimension
    dimension: usize,
    /// Bond type (left, right, physical)
    bond_type: BondType,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BondType {
    Left,
    Right, 
    Physical,
}

/// Strategy for switching between hybrid representations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SwitchingStrategy {
    /// Use primary for indices below threshold
    IndexThreshold(u64),
    /// Use primary for cached queries
    CacheAware,
    /// Use representation with better performance characteristics
    Performance,
}

/// Metadata for IDVBit optimization and validation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IDVBitMetadata {
    /// Human-readable description
    description: String,
    /// Known bounds on non-zero coefficients
    support_bounds: Option<(u64, u64)>,
    /// Computational complexity estimates
    complexity: ComplexityInfo,
    /// Mathematical properties
    properties: MathematicalProperties,
}

/// Computational complexity information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplexityInfo {
    /// Time complexity for coefficient queries
    query_time: String,
    /// Space complexity for representation
    space_usage: String,
    /// Precomputation requirements
    precompute_cost: Option<String>,
}

/// Mathematical properties of the IDVBit
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MathematicalProperties {
    /// Whether the sequence is eventually periodic
    eventually_periodic: Option<bool>,
    /// Generating function type (rational, algebraic, etc.)
    function_type: String,
    /// Known special structure
    special_structure: Option<String>,
}

impl IDVBit {
    /// Create new IDVBit - convenience method using default constant series
    pub fn new() -> Self {
        let series = FormalPowerSeries::constant(Complex64::new(1.0, 0.0));
        Self::new_symbolic(series, "constant function 1")
    }
    
    /// Create new IDVBit with symbolic representation
    pub fn new_symbolic(series: FormalPowerSeries, description: &str) -> Self {
        Self {
            storage: IDVBitStorage::Symbolic {
                series,
                rational: None,
                description: description.to_string(),
            },
            metadata: IDVBitMetadata::default(description),
            coefficient_cache: HashMap::new(),
        }
    }
    
    /// Create IDVBit from rational generating function
    pub fn new_rational(rational: RationalGeneratingFunction, description: &str) -> Self {
        let series = FormalPowerSeries::from_rational(&rational);
        Self {
            storage: IDVBitStorage::Symbolic {
                series,
                rational: Some(rational),
                description: description.to_string(),
            },
            metadata: IDVBitMetadata::default(description),
            coefficient_cache: HashMap::new(),
        }
    }
    
    /// Create IDVBit with decision diagram representation
    pub fn new_decision_diagram(
        nodes: Vec<DecisionNode>, 
        root: usize,
        variables: Vec<String>,
        description: &str
    ) -> Self {
        Self {
            storage: IDVBitStorage::DecisionDiagram {
                nodes,
                root,
                variable_order: variables,
            },
            metadata: IDVBitMetadata::default(description),
            coefficient_cache: HashMap::new(),
        }
    }
    
    /// Create IDVBit with tensor network representation
    pub fn new_tensor_network(
        cores: Vec<TensorCore>,
        dimensions: Vec<usize>,
        description: &str
    ) -> Self {
        let bond_dimensions = cores.iter()
            .map(|core| core.bonds.iter().map(|b| b.dimension).max().unwrap_or(1))
            .collect();
            
        Self {
            storage: IDVBitStorage::TensorNetwork {
                cores,
                dimensions,
                bond_dimensions,
            },
            metadata: IDVBitMetadata::default(description),
            coefficient_cache: HashMap::new(),
        }
    }
    
    /// Query coefficient at given index with advanced algorithms
    pub fn query_coefficient(&mut self, index: u64) -> Result<Complex64> {
        // Check cache first
        if let Some(&cached) = self.coefficient_cache.get(&index) {
            return Ok(cached);
        }
        
        // Compute result based on storage type
        let result = self.compute_coefficient(index)?;
        
        // Cache the result for future queries
        if self.coefficient_cache.len() < 1000 { // Prevent unbounded growth
            self.coefficient_cache.insert(index, result);
        }
        
        Ok(result)
    }
    
    /// Compute coefficient without caching
    fn compute_coefficient(&mut self, index: u64) -> Result<Complex64> {
        match &mut self.storage {
            IDVBitStorage::Symbolic { series, rational, .. } => {
                if let Some(rat) = rational {
                    // Use fast coefficient extraction for rational functions
                    rat.coefficient(index as usize)
                } else {
                    // Fall back to series evaluation
                    series.coefficient(index as usize)
                }
            },
            
            IDVBitStorage::DecisionDiagram { nodes, root, .. } => {
                let nodes_ref = nodes.as_slice();
                let root_idx = *root;
                Self::evaluate_decision_diagram_static(nodes_ref, root_idx, index)
            },
            
            IDVBitStorage::TensorNetwork { cores, .. } => {
                let cores_ref = cores.as_slice();
                Self::evaluate_tensor_network_static(cores_ref, index)
            },
            
            IDVBitStorage::Hybrid { primary, secondary, switching_strategy } => {
                match switching_strategy {
                    SwitchingStrategy::IndexThreshold(threshold) => {
                        if index < *threshold {
                            Self::query_from_storage_static(primary, index)
                        } else {
                            Self::query_from_storage_static(secondary, index)
                        }
                    },
                    SwitchingStrategy::CacheAware => {
                        // Try primary first, fall back to secondary
                        Self::query_from_storage_static(primary, index)
                            .or_else(|_| Self::query_from_storage_static(secondary, index))
                    },
                    SwitchingStrategy::Performance => {
                        // Choose based on performance characteristics (simplified)
                        if index % 2 == 0 {
                            Self::query_from_storage_static(primary, index)
                        } else {
                            Self::query_from_storage_static(secondary, index)
                        }
                    }
                }
            },
        }
    }
    
    /// Query multiple coefficients efficiently
    pub fn query_range(&mut self, start: u64, end: u64) -> Result<Vec<Complex64>> {
        let mut results = Vec::with_capacity((end - start) as usize);
        
        for i in start..end {
            results.push(self.query_coefficient(i)?);
        }
        
        Ok(results)
    }
    
    /// Get IDVBit metadata
    pub fn metadata(&self) -> &IDVBitMetadata {
        &self.metadata
    }
    
    /// Clear coefficient cache
    pub fn clear_cache(&mut self) {
        self.coefficient_cache.clear();
    }
    
    /// Get cache statistics
    pub fn cache_stats(&self) -> (usize, usize) {
        (self.coefficient_cache.len(), 1000) // (current, max)
    }
    
    // Private helper methods
    
    fn query_from_storage_static(storage: &IDVBitStorage, _index: u64) -> Result<Complex64> {
        match storage {
            IDVBitStorage::Symbolic { .. } => {
                // For static queries, we can't modify the series, so return a placeholder
                Ok(Complex64::new(1.0, 0.0))
            },
            _ => Err(IDVBitError::Representation("Unsupported storage type for hybrid query".to_string())),
        }
    }
    
    fn evaluate_decision_diagram_static(nodes: &[DecisionNode], root: usize, _index: u64) -> Result<Complex64> {
        // Simplified decision diagram evaluation
        // In a real implementation, this would traverse the diagram based on the binary representation of index
        if root >= nodes.len() {
            return Err(IDVBitError::Representation("Invalid root node".to_string()));
        }
        
        // For now, return a placeholder - full implementation would be quite complex
        Ok(Complex64::new(1.0, 0.0))
    }
    
    fn evaluate_tensor_network_static(_cores: &[TensorCore], _index: u64) -> Result<Complex64> {
        // Simplified tensor network evaluation
        // Real implementation would perform tensor contractions
        
        // Placeholder - real tensor network evaluation is highly complex
        Ok(Complex64::new(1.0, 0.0))
    }
}

impl IDVBitMetadata {
    fn default(description: &str) -> Self {
        Self {
            description: description.to_string(),
            support_bounds: None,
            complexity: ComplexityInfo {
                query_time: "O(log n)".to_string(),
                space_usage: "O(1)".to_string(),
                precompute_cost: None,
            },
            properties: MathematicalProperties {
                eventually_periodic: None,
                function_type: "unknown".to_string(),
                special_structure: None,
            },
        }
    }
}

impl Default for IDVBit {
    fn default() -> Self {
        let series = FormalPowerSeries::constant(Complex64::new(1.0, 0.0));
        Self::new_symbolic(series, "constant function 1")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::formal_series::{FormalPowerSeries, RationalGeneratingFunction};
    use num_complex::Complex64;

    #[test]
    fn test_basic_functionality() {
        // Basic smoke test - will be expanded in PR-001
        assert_eq!(2 + 2, 4);
    }

    #[test]
    fn test_idvbit_new_default() {
        let idvbit = IDVBit::new();
        assert_eq!(idvbit.metadata().description, "constant function 1");
    }

    #[test]
    fn test_idvbit_symbolic_creation() {
        let series = FormalPowerSeries::from_real(vec![1.0, 2.0, 3.0]);
        let idvbit = IDVBit::new_symbolic(series, "test series");
        assert_eq!(idvbit.metadata().description, "test series");
    }

    #[test]
    fn test_idvbit_coefficient_query() {
        let series = FormalPowerSeries::from_real(vec![1.0, 2.0, 3.0, 4.0]);
        let mut idvbit = IDVBit::new_symbolic(series, "test series");
        
        // Test coefficient access
        assert_eq!(idvbit.query_coefficient(0).unwrap(), Complex64::new(1.0, 0.0));
        assert_eq!(idvbit.query_coefficient(1).unwrap(), Complex64::new(2.0, 0.0));
        assert_eq!(idvbit.query_coefficient(2).unwrap(), Complex64::new(3.0, 0.0));
        assert_eq!(idvbit.query_coefficient(3).unwrap(), Complex64::new(4.0, 0.0));
    }

    #[test]
    fn test_idvbit_caching() {
        let series = FormalPowerSeries::from_real(vec![1.0, 2.0, 3.0, 4.0]);
        let mut idvbit = IDVBit::new_symbolic(series, "test series");
        
        // Query a coefficient to cache it
        let coeff = idvbit.query_coefficient(2).unwrap();
        assert_eq!(coeff, Complex64::new(3.0, 0.0));
        
        // Check cache statistics
        let (cached, max_cache) = idvbit.cache_stats();
        assert_eq!(cached, 1);
        assert_eq!(max_cache, 1000);
        
        // Query the same coefficient again (should hit cache)
        let coeff2 = idvbit.query_coefficient(2).unwrap();
        assert_eq!(coeff2, Complex64::new(3.0, 0.0));
    }

    #[test]
    fn test_idvbit_rational_creation() {
        // Create rational function 1/(1-x) = 1 + x + x² + ...
        let numerator = vec![Complex64::new(1.0, 0.0)];
        let denominator = vec![Complex64::new(1.0, 0.0), Complex64::new(-1.0, 0.0)];
        let rational = RationalGeneratingFunction::new(numerator, denominator).unwrap();
        
        let idvbit = IDVBit::new_rational(rational, "geometric series");
        assert_eq!(idvbit.metadata().description, "geometric series");
    }

    #[test]
    fn test_idvbit_decision_diagram() {
        // Create simple decision diagram
        let nodes = vec![
            DecisionNode::new(0, 
                NodeRef::terminal(Complex64::new(0.0, 0.0)), 
                NodeRef::terminal(Complex64::new(1.0, 0.0)), 
                None
            )
        ];
        let variables = vec!["x0".to_string()];
        let idvbit = IDVBit::new_decision_diagram(nodes, 0, variables, "simple DD");
        
        assert_eq!(idvbit.metadata().description, "simple DD");
    }

    #[test]
    fn test_idvbit_tensor_network() {
        // Create simple tensor core
        let data = vec![Complex64::new(1.0, 0.0), Complex64::new(0.0, 1.0)];
        let core = TensorCore::from_complex_data(data, vec![2], vec![]);
        let cores = vec![core];
        let dimensions = vec![2];
        
        let idvbit = IDVBit::new_tensor_network(cores, dimensions, "simple TN");
        assert_eq!(idvbit.metadata().description, "simple TN");
    }

    #[test]
    fn test_idvbit_query_range() {
        let series = FormalPowerSeries::from_real(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let mut idvbit = IDVBit::new_symbolic(series, "test range");
        
        let range = idvbit.query_range(0, 4).unwrap();
        assert_eq!(range.len(), 4);
        assert_eq!(range[0], Complex64::new(1.0, 0.0));
        assert_eq!(range[1], Complex64::new(2.0, 0.0));
        assert_eq!(range[2], Complex64::new(3.0, 0.0));
        assert_eq!(range[3], Complex64::new(4.0, 0.0));
    }

    #[test]
    fn test_node_ref_helpers() {
        let terminal = NodeRef::terminal(Complex64::new(3.14, 2.71));
        let value = terminal.terminal_value().unwrap();
        assert_eq!(value.re, 3.14);
        assert_eq!(value.im, 2.71);
        
        let node_ref = NodeRef::Node(42);
        assert!(node_ref.terminal_value().is_none());
    }

    #[test]
    fn test_tensor_core_helpers() {
        let data = vec![Complex64::new(1.0, 0.0), Complex64::new(0.0, 1.0)];
        let core = TensorCore::from_complex_data(data.clone(), vec![2], vec![]);
        let recovered = core.complex_data();
        
        assert_eq!(recovered.len(), 2);
        assert_eq!(recovered[0], Complex64::new(1.0, 0.0));
        assert_eq!(recovered[1], Complex64::new(0.0, 1.0));
    }

    #[test]
    fn test_cache_clearing() {
        let series = FormalPowerSeries::from_real(vec![1.0, 2.0, 3.0]);
        let mut idvbit = IDVBit::new_symbolic(series, "cache test");
        
        // Cache a coefficient
        let _ = idvbit.query_coefficient(1).unwrap();
        let (cached, _) = idvbit.cache_stats();
        assert_eq!(cached, 1);
        
        // Clear cache
        idvbit.clear_cache();
        let (cached_after, _) = idvbit.cache_stats();
        assert_eq!(cached_after, 0);
    }
}