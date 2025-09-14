//! Problem mapping utilities for God-Index implementation
//!
//! Maps structured problem instances to mathematical representations suitable
//! for coefficient extraction from IDVBit generating functions.

use std::collections::HashMap;
use idvbit_core::{IDVBit, Result as IDVBitResult};
use super::{GodIndexResult, GodIndex};

/// Problem classification for specialized mappings
#[derive(Debug, Clone, PartialEq)]
pub enum ProblemClass {
    /// Boolean satisfiability problems (CNF formulas)
    BooleanSat,
    /// Subset sum with bounded weights
    SubsetSum,
    /// Graph coloring for small graphs
    GraphColoring,
    /// Knapsack with small capacity
    Knapsack,
    /// General combinatorial enumeration
    Combinatorial,
}

/// Subset sum problem instance
#[derive(Debug, Clone)]
pub struct SubsetSumProblem {
    pub weights: Vec<u32>,
    pub target: u32,
}

/// Boolean SAT problem instance (simplified CNF)
#[derive(Debug, Clone)]
pub struct SatProblem {
    pub variables: usize,
    pub clauses: Vec<Vec<i32>>, // positive for var, negative for negation
}

/// Graph coloring problem instance
#[derive(Debug, Clone)]
pub struct GraphColoringProblem {
    pub vertices: usize,
    pub edges: Vec<(usize, usize)>,
    pub colors: usize,
}

/// Problem mapper that classifies instances and generates IDVBit mappings
#[derive(Debug)]
pub struct ProblemMapper {
    cache: HashMap<String, GodIndexResult>,
}

impl Default for ProblemMapper {
    fn default() -> Self {
        Self::new()
    }
}

impl ProblemMapper {
    pub fn new() -> Self {
        Self {
            cache: HashMap::new(),
        }
    }

    /// Classify a problem based on its structure
    pub fn classify_problem(&self, problem_desc: &str) -> ProblemClass {
        if problem_desc.contains("CNF") || problem_desc.contains("SAT") {
            ProblemClass::BooleanSat
        } else if problem_desc.contains("subset") || problem_desc.contains("sum") {
            ProblemClass::SubsetSum
        } else if problem_desc.contains("coloring") || problem_desc.contains("graph") {
            ProblemClass::GraphColoring
        } else if problem_desc.contains("knapsack") {
            ProblemClass::Knapsack
        } else {
            ProblemClass::Combinatorial
        }
    }

    /// Map subset sum to generating function representation
    pub fn map_subset_sum(&mut self, problem: &SubsetSumProblem) -> IDVBitResult<GodIndexResult> {
        let cache_key = format!("subset_sum_{:?}_{}", problem.weights, problem.target);
        
        if let Some(cached) = self.cache.get(&cache_key) {
            return Ok(cached.clone());
        }

        // For subset sum, create generating function Π(1 + x^w_i) for weights w_i
        // Coefficient of x^target gives number of solutions
        if problem.weights.len() > 20 {
            // Large problems require compilation
            let result = GodIndexResult::CompilationRequired { 
                problem_class: "SubsetSum".to_string() 
            };
            self.cache.insert(cache_key, result.clone());
            return Ok(result);
        }

        // Small instances: direct coefficient extraction
        let total_sum: u32 = problem.weights.iter().sum();
        if problem.target > total_sum {
            let result = GodIndexResult::DirectIndex(0); // No solutions
            self.cache.insert(cache_key, result.clone());
            return Ok(result);
        }

        // Create polynomial representation
        // For generating function approach, we need the coefficient of x^target in Π(1 + x^w_i)
        let result = GodIndexResult::Parameterized { 
            params: problem.weights.iter().map(|&w| w as f64).collect()
        };
        
        self.cache.insert(cache_key, result.clone());
        Ok(result)
    }

    /// Map Boolean SAT to generating function (small instances only)
    pub fn map_boolean_sat(&mut self, problem: &SatProblem) -> IDVBitResult<GodIndexResult> {
        let cache_key = format!("sat_{}_{:?}", problem.variables, problem.clauses.len());
        
        if let Some(cached) = self.cache.get(&cache_key) {
            return Ok(cached.clone());
        }

        if problem.variables > 16 {
            // Large SAT instances require knowledge compilation
            let result = GodIndexResult::CompilationRequired { 
                problem_class: "BooleanSAT".to_string() 
            };
            self.cache.insert(cache_key, result.clone());
            return Ok(result);
        }

        // For small SAT instances, enumerate satisfying assignments
        let satisfying_assignments = self.count_sat_solutions(problem);
        
        let result = GodIndexResult::DirectIndex(satisfying_assignments);
        self.cache.insert(cache_key, result.clone());
        Ok(result)
    }

    /// Count satisfying assignments for small SAT instances (brute force enumeration)
    fn count_sat_solutions(&self, problem: &SatProblem) -> u64 {
        let mut count = 0;
        let total_assignments = 1u64 << problem.variables;
        
        for assignment in 0..total_assignments {
            if self.check_sat_assignment(problem, assignment) {
                count += 1;
            }
        }
        
        count
    }

    /// Check if a truth assignment satisfies all clauses
    fn check_sat_assignment(&self, problem: &SatProblem, assignment: u64) -> bool {
        for clause in &problem.clauses {
            let mut clause_satisfied = false;
            
            for &literal in clause {
                let var_index = (literal.abs() - 1) as u32;
                let var_value = (assignment >> var_index) & 1 == 1;
                let literal_satisfied = if literal > 0 { var_value } else { !var_value };
                
                if literal_satisfied {
                    clause_satisfied = true;
                    break;
                }
            }
            
            if !clause_satisfied {
                return false;
            }
        }
        
        true
    }

    /// Map graph coloring to generating function (small graphs only)
    pub fn map_graph_coloring(&mut self, problem: &GraphColoringProblem) -> IDVBitResult<GodIndexResult> {
        let cache_key = format!("coloring_{}_{:?}_{}", problem.vertices, problem.edges.len(), problem.colors);
        
        if let Some(cached) = self.cache.get(&cache_key) {
            return Ok(cached.clone());
        }

        if problem.vertices > 10 {
            let result = GodIndexResult::CompilationRequired { 
                problem_class: "GraphColoring".to_string() 
            };
            self.cache.insert(cache_key, result.clone());
            return Ok(result);
        }

        // Count valid colorings by enumeration
        let valid_colorings = self.count_graph_colorings(problem);
        
        let result = GodIndexResult::DirectIndex(valid_colorings);
        self.cache.insert(cache_key, result.clone());
        Ok(result)
    }

    /// Count valid graph colorings (brute force for small graphs)
    fn count_graph_colorings(&self, problem: &GraphColoringProblem) -> u64 {
        let mut count = 0;
        let total_colorings = (problem.colors as u64).pow(problem.vertices as u32);
        
        for coloring in 0..total_colorings {
            if self.check_valid_coloring(problem, coloring) {
                count += 1;
            }
        }
        
        count
    }

    /// Check if a coloring is valid (no adjacent vertices have same color)
    fn check_valid_coloring(&self, problem: &GraphColoringProblem, coloring: u64) -> bool {
        // Extract color assignment for each vertex
        let mut colors = vec![0; problem.vertices];
        let mut temp_coloring = coloring;
        
        for i in 0..problem.vertices {
            colors[i] = (temp_coloring % problem.colors as u64) as usize;
            temp_coloring /= problem.colors as u64;
        }
        
        // Check all edges
        for &(u, v) in &problem.edges {
            if colors[u] == colors[v] {
                return false;
            }
        }
        
        true
    }
}

/// God-Index implementation for subset sum problems
impl GodIndex<SubsetSumProblem> for ProblemMapper {
    fn map_problem(&self, problem: &SubsetSumProblem) -> IDVBitResult<GodIndexResult> {
        // Create a mutable copy to use the cache
        let mut mapper = ProblemMapper {
            cache: HashMap::new(),
        };
        mapper.map_subset_sum(problem)
    }

    /// Extract solution from IDVBit given mapping result
    fn extract_solution(&self, idvbit: &mut IDVBit, mapping: &GodIndexResult) -> IDVBitResult<Vec<u8>> {
        match mapping {
            GodIndexResult::DirectIndex(count) => {
                // Return binary encoding of solution count
                Ok(count.to_le_bytes().to_vec())
            }
            GodIndexResult::Parameterized { params } => {
                // For subset sum, use coefficient extraction
                // This is a placeholder - full implementation would extract actual subsets
                let coefficient = idvbit.query_coefficient(params[0] as u64)?;
                Ok(coefficient.to_string().as_bytes().to_vec())
            }
            _ => Ok(vec![0]), // Placeholder
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_problem_classification() {
        let mapper = ProblemMapper::new();
        
        assert_eq!(mapper.classify_problem("CNF formula"), ProblemClass::BooleanSat);
        assert_eq!(mapper.classify_problem("subset sum problem"), ProblemClass::SubsetSum);
        assert_eq!(mapper.classify_problem("graph coloring"), ProblemClass::GraphColoring);
        assert_eq!(mapper.classify_problem("knapsack optimization"), ProblemClass::Knapsack);
    }

    #[test]
    fn test_subset_sum_small() {
        let mut mapper = ProblemMapper::new();
        let problem = SubsetSumProblem {
            weights: vec![1, 2, 3],
            target: 4,
        };
        
        let result = mapper.map_subset_sum(&problem).unwrap();
        match result {
            GodIndexResult::Parameterized { params } => {
                assert_eq!(params, vec![1.0, 2.0, 3.0]);
            }
            _ => panic!("Expected parameterized result"),
        }
    }

    #[test]
    fn test_subset_sum_impossible() {
        let mut mapper = ProblemMapper::new();
        let problem = SubsetSumProblem {
            weights: vec![1, 2],
            target: 10, // Impossible target
        };
        
        let result = mapper.map_subset_sum(&problem).unwrap();
        match result {
            GodIndexResult::DirectIndex(0) => {}, // No solutions
            _ => panic!("Expected direct index 0 for impossible target"),
        }
    }

    #[test]
    fn test_boolean_sat_small() {
        let mut mapper = ProblemMapper::new();
        let problem = SatProblem {
            variables: 2,
            clauses: vec![vec![1], vec![-2]], // x1 AND NOT x2
        };
        
        let result = mapper.map_boolean_sat(&problem).unwrap();
        match result {
            GodIndexResult::DirectIndex(1) => {}, // One satisfying assignment: x1=true, x2=false
            _ => panic!("Expected direct index 1"),
        }
    }

    #[test]
    fn test_sat_solution_counting() {
        let mapper = ProblemMapper::new();
        let problem = SatProblem {
            variables: 2,
            clauses: vec![vec![1, 2]], // x1 OR x2
        };
        
        let count = mapper.count_sat_solutions(&problem);
        assert_eq!(count, 3); // Three satisfying assignments: (1,0), (0,1), (1,1)
    }

    #[test]
    fn test_graph_coloring_small() {
        let mut mapper = ProblemMapper::new();
        let problem = GraphColoringProblem {
            vertices: 3,
            edges: vec![(0, 1), (1, 2)], // Path graph
            colors: 2,
        };
        
        let result = mapper.map_graph_coloring(&problem).unwrap();
        match result {
            GodIndexResult::DirectIndex(count) => {
                assert!(count > 0); // Should have valid colorings
            }
            _ => panic!("Expected direct index result"),
        }
    }

    #[test]
    fn test_caching() {
        let mut mapper = ProblemMapper::new();
        let problem = SubsetSumProblem {
            weights: vec![1, 2],
            target: 3,
        };
        
        // First call
        let result1 = mapper.map_subset_sum(&problem).unwrap();
        
        // Second call should use cache
        let result2 = mapper.map_subset_sum(&problem).unwrap();
        
        // Results should be identical
        match (result1, result2) {
            (GodIndexResult::Parameterized { params: p1 }, GodIndexResult::Parameterized { params: p2 }) => {
                assert_eq!(p1, p2);
            }
            _ => panic!("Expected parameterized results"),
        }
    }
}