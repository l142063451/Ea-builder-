//! Solution verification utilities for God-Index implementation
//!
//! Verifies that extracted solutions are mathematically correct and
//! provides certificate-based validation for problem instances.

use super::problem_mapping::*;
use std::collections::HashMap;

/// Solution verifier for God-Index extracted solutions
#[derive(Debug)]
pub struct Verifier {
    /// Cache for verification results
    verification_cache: HashMap<String, bool>,
}

impl Default for Verifier {
    fn default() -> Self {
        Self::new()
    }
}

impl Verifier {
    pub fn new() -> Self {
        Self {
            verification_cache: HashMap::new(),
        }
    }

    /// Verify a subset sum solution
    pub fn verify_subset_sum(&mut self, problem: &SubsetSumProblem, solution: &[usize]) -> bool {
        let cache_key = format!("subset_sum_{:?}_{:?}", problem.weights, solution);
        
        if let Some(&cached) = self.verification_cache.get(&cache_key) {
            return cached;
        }

        // Check that all indices are valid
        for &index in solution {
            if index >= problem.weights.len() {
                self.verification_cache.insert(cache_key, false);
                return false;
            }
        }

        // Check that sum equals target
        let sum: u32 = solution.iter().map(|&i| problem.weights[i]).sum();
        let is_valid = sum == problem.target;
        
        self.verification_cache.insert(cache_key, is_valid);
        is_valid
    }

    /// Verify a Boolean SAT solution (truth assignment)
    pub fn verify_boolean_sat(&mut self, problem: &SatProblem, assignment: u64) -> bool {
        let cache_key = format!("sat_{}_{:?}_{}", problem.variables, problem.clauses, assignment);
        
        if let Some(&cached) = self.verification_cache.get(&cache_key) {
            return cached;
        }

        // Check assignment is within valid range
        if assignment >= (1u64 << problem.variables) {
            self.verification_cache.insert(cache_key, false);
            return false;
        }

        // Check each clause
        for clause in &problem.clauses {
            let mut clause_satisfied = false;
            
            for &literal in clause {
                let var_index = (literal.abs() - 1) as u32;
                if var_index >= problem.variables as u32 {
                    self.verification_cache.insert(cache_key, false);
                    return false;
                }
                
                let var_value = (assignment >> var_index) & 1 == 1;
                let literal_satisfied = if literal > 0 { var_value } else { !var_value };
                
                if literal_satisfied {
                    clause_satisfied = true;
                    break;
                }
            }
            
            if !clause_satisfied {
                self.verification_cache.insert(cache_key, false);
                return false;
            }
        }

        self.verification_cache.insert(cache_key, true);
        true
    }

    /// Verify a graph coloring solution
    pub fn verify_graph_coloring(&mut self, problem: &GraphColoringProblem, coloring: &[usize]) -> bool {
        let cache_key = format!("coloring_{}_{}_{:?}", problem.vertices, problem.edges.len(), coloring);
        
        if let Some(&cached) = self.verification_cache.get(&cache_key) {
            return cached;
        }

        // Check coloring has correct length
        if coloring.len() != problem.vertices {
            self.verification_cache.insert(cache_key, false);
            return false;
        }

        // Check all colors are valid
        for &color in coloring {
            if color >= problem.colors {
                self.verification_cache.insert(cache_key, false);
                return false;
            }
        }

        // Check no adjacent vertices have same color
        for &(u, v) in &problem.edges {
            if u >= problem.vertices || v >= problem.vertices {
                self.verification_cache.insert(cache_key, false);
                return false;
            }
            
            if coloring[u] == coloring[v] {
                self.verification_cache.insert(cache_key, false);
                return false;
            }
        }

        self.verification_cache.insert(cache_key, true);
        true
    }

    /// Verify multiple solutions for a problem
    pub fn verify_all_subset_sum_solutions(&mut self, problem: &SubsetSumProblem, solutions: &[Vec<usize>]) -> Vec<bool> {
        solutions.iter()
            .map(|solution| self.verify_subset_sum(problem, solution))
            .collect()
    }

    /// Verify multiple SAT assignments
    pub fn verify_all_sat_assignments(&mut self, problem: &SatProblem, assignments: &[u64]) -> Vec<bool> {
        assignments.iter()
            .map(|&assignment| self.verify_boolean_sat(problem, assignment))
            .collect()
    }

    /// Generate certificate for solution verification
    pub fn generate_certificate(&self, problem_type: &str, problem_data: &str, solution: &str) -> String {
        format!("CERTIFICATE:{}:{}:{}", problem_type, problem_data, solution)
    }

    /// Verify a certificate
    pub fn verify_certificate(&self, certificate: &str) -> bool {
        // Basic certificate format validation
        let parts: Vec<&str> = certificate.split(':').collect();
        parts.len() >= 4 && parts[0] == "CERTIFICATE"
    }

    /// Batch verification for performance testing
    pub fn batch_verify_subset_sum(&mut self, problems: &[SubsetSumProblem], solutions: &[Vec<usize>]) -> Vec<bool> {
        if problems.len() != solutions.len() {
            return vec![false; problems.len()];
        }

        problems.iter()
            .zip(solutions.iter())
            .map(|(problem, solution)| self.verify_subset_sum(problem, solution))
            .collect()
    }

    /// Statistical verification report
    pub fn generate_verification_report(&self, results: &[bool]) -> HashMap<String, f64> {
        let mut report = HashMap::new();
        
        let total = results.len() as f64;
        let valid = results.iter().filter(|&&x| x).count() as f64;
        let invalid = total - valid;
        
        report.insert("total_verifications".to_string(), total);
        report.insert("valid_solutions".to_string(), valid);
        report.insert("invalid_solutions".to_string(), invalid);
        report.insert("success_rate".to_string(), if total > 0.0 { valid / total } else { 0.0 });
        
        report
    }

    /// Clear verification cache
    pub fn clear_cache(&mut self) {
        self.verification_cache.clear();
    }

    /// Get cache statistics
    pub fn cache_stats(&self) -> HashMap<String, usize> {
        let mut stats = HashMap::new();
        stats.insert("cache_size".to_string(), self.verification_cache.len());
        stats.insert("cached_verifications".to_string(), self.verification_cache.len());
        stats
    }
}

/// Verification result with detailed information
#[derive(Debug, Clone)]
pub struct VerificationResult {
    pub is_valid: bool,
    pub error_message: Option<String>,
    pub certificate: Option<String>,
    pub verification_time_ms: Option<u64>,
}

impl VerificationResult {
    pub fn valid() -> Self {
        Self {
            is_valid: true,
            error_message: None,
            certificate: None,
            verification_time_ms: None,
        }
    }

    pub fn invalid(message: &str) -> Self {
        Self {
            is_valid: false,
            error_message: Some(message.to_string()),
            certificate: None,
            verification_time_ms: None,
        }
    }

    pub fn with_certificate(mut self, cert: String) -> Self {
        self.certificate = Some(cert);
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_subset_sum_verification_valid() {
        let mut verifier = Verifier::new();
        let problem = SubsetSumProblem {
            weights: vec![1, 2, 3, 4],
            target: 5,
        };
        
        // Valid solution: indices 0 and 2 (weights 1 + 4 = 5)
        let solution = vec![0, 3]; // weights[0] + weights[3] = 1 + 4 = 5
        assert!(verifier.verify_subset_sum(&problem, &solution));
        
        // Another valid solution: index 1 and 2 (weights 2 + 3 = 5)
        let solution2 = vec![1, 2]; // weights[1] + weights[2] = 2 + 3 = 5
        assert!(verifier.verify_subset_sum(&problem, &solution2));
    }

    #[test]
    fn test_subset_sum_verification_invalid() {
        let mut verifier = Verifier::new();
        let problem = SubsetSumProblem {
            weights: vec![1, 2, 3, 4],
            target: 5,
        };
        
        // Invalid solution: wrong sum
        let solution = vec![0, 1]; // weights[0] + weights[1] = 1 + 2 = 3 ≠ 5
        assert!(!verifier.verify_subset_sum(&problem, &solution));
        
        // Invalid solution: out of bounds index
        let solution2 = vec![0, 10]; // index 10 doesn't exist
        assert!(!verifier.verify_subset_sum(&problem, &solution2));
    }

    #[test]
    fn test_boolean_sat_verification() {
        let mut verifier = Verifier::new();
        let problem = SatProblem {
            variables: 3,
            clauses: vec![vec![1, 2], vec![-1, 3], vec![-2, -3]], // (x1 ∨ x2) ∧ (¬x1 ∨ x3) ∧ (¬x2 ∨ ¬x3)
        };
        
        // Test satisfying assignment: x1=true, x2=false, x3=true (binary 101 = decimal 5)
        assert!(verifier.verify_boolean_sat(&problem, 5));
        
        // Test unsatisfying assignment: x1=true, x2=true, x3=true (binary 111 = decimal 7)
        // This violates clause 3: (¬x2 ∨ ¬x3) since both x2 and x3 are true
        assert!(!verifier.verify_boolean_sat(&problem, 7));
    }

    #[test]
    fn test_graph_coloring_verification() {
        let mut verifier = Verifier::new();
        let problem = GraphColoringProblem {
            vertices: 3,
            edges: vec![(0, 1), (1, 2)], // Path graph: 0-1-2
            colors: 2,
        };
        
        // Valid coloring: alternating colors
        let coloring = vec![0, 1, 0]; // vertex 0: color 0, vertex 1: color 1, vertex 2: color 0
        assert!(verifier.verify_graph_coloring(&problem, &coloring));
        
        // Invalid coloring: adjacent vertices same color
        let invalid_coloring = vec![0, 0, 1]; // vertices 0 and 1 both have color 0
        assert!(!verifier.verify_graph_coloring(&problem, &invalid_coloring));
        
        // Invalid coloring: out of range color
        let invalid_coloring2 = vec![0, 1, 3]; // color 3 doesn't exist (only colors 0,1)
        assert!(!verifier.verify_graph_coloring(&problem, &invalid_coloring2));
    }

    #[test]
    fn test_verification_caching() {
        let mut verifier = Verifier::new();
        let problem = SubsetSumProblem {
            weights: vec![1, 2, 3],
            target: 3,
        };
        let solution = vec![2]; // weight[2] = 3
        
        // First verification
        let result1 = verifier.verify_subset_sum(&problem, &solution);
        
        // Second verification should use cache
        let result2 = verifier.verify_subset_sum(&problem, &solution);
        
        assert_eq!(result1, result2);
        assert!(result1); // Should be valid
        
        // Check cache stats
        let stats = verifier.cache_stats();
        assert!(stats.get("cache_size").unwrap() > &0);
    }

    #[test]
    fn test_batch_verification() {
        let mut verifier = Verifier::new();
        let problems = vec![
            SubsetSumProblem { weights: vec![1, 2], target: 3 },
            SubsetSumProblem { weights: vec![2, 3], target: 5 },
        ];
        let solutions = vec![
            vec![0, 1], // 1 + 2 = 3 ✓
            vec![0, 1], // 2 + 3 = 5 ✓
        ];
        
        let results = verifier.batch_verify_subset_sum(&problems, &solutions);
        assert_eq!(results, vec![true, true]);
    }

    #[test]
    fn test_verification_report() {
        let verifier = Verifier::new();
        let results = vec![true, true, false, true, false];
        let report = verifier.generate_verification_report(&results);
        
        assert_eq!(report.get("total_verifications").unwrap(), &5.0);
        assert_eq!(report.get("valid_solutions").unwrap(), &3.0);
        assert_eq!(report.get("invalid_solutions").unwrap(), &2.0);
        assert_eq!(report.get("success_rate").unwrap(), &0.6);
    }

    #[test]
    fn test_certificate_generation() {
        let verifier = Verifier::new();
        let cert = verifier.generate_certificate("SubsetSum", "weights=[1,2,3],target=3", "solution=[2]");
        
        assert!(cert.starts_with("CERTIFICATE:"));
        assert!(verifier.verify_certificate(&cert));
        
        // Invalid certificate
        assert!(!verifier.verify_certificate("INVALID_CERT"));
    }
}