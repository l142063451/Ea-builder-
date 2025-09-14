//! Index extraction utilities for God-Index implementation
//!
//! Extracts coefficients and solutions from IDVBit representations using
//! the mathematical foundations established in PR-001 and PR-002.

use idvbit_core::{IDVBit, FormalPowerSeries, Result as IDVBitResult, IDVBitError};
use super::problem_mapping::*;
use rug::Complex;
use std::collections::HashMap;

/// Index extractor that connects God-Index mappings to IDVBit coefficient extraction
#[derive(Debug)]
pub struct IndexExtractor {
    /// Cache for generated IDVBits per problem class
    idvbit_cache: HashMap<String, IDVBit>,
    /// Cache for extracted coefficients
    coefficient_cache: HashMap<(String, u64), Complex>,
}

impl Default for IndexExtractor {
    fn default() -> Self {
        Self::new()
    }
}

impl IndexExtractor {
    pub fn new() -> Self {
        Self {
            idvbit_cache: HashMap::new(),
            coefficient_cache: HashMap::new(),
        }
    }

    /// Extract coefficient from generating function at given index
    pub fn extract_coefficient(&mut self, idvbit: &mut IDVBit, index: u64) -> IDVBitResult<Complex> {
        let cache_key = (format!("{:?}", idvbit), index);
        
        if let Some(cached) = self.coefficient_cache.get(&cache_key) {
            return Ok(cached.clone());
        }

        let coefficient = idvbit.query_coefficient(index)?;
        // Convert from num_complex::Complex64 to rug::Complex
        let rug_complex = Complex::with_val(53, (coefficient.re, coefficient.im));
        self.coefficient_cache.insert(cache_key, rug_complex.clone());
        
        Ok(rug_complex)
    }

    /// Create IDVBit for subset sum problem using generating function approach
    pub fn create_subset_sum_idvbit(&mut self, problem: &SubsetSumProblem) -> IDVBitResult<IDVBit> {
        let cache_key = format!("subset_sum_{:?}", problem.weights);
        
        if let Some(cached) = self.idvbit_cache.get(&cache_key) {
            return Ok(cached.clone());
        }

        // Create generating function Π(1 + x^w_i) for subset sum
        // Start with polynomial 1 + x^w_0
        use num_complex::Complex64;
        let mut current_poly = vec![Complex64::new(0.0, 0.0); problem.weights[0] as usize + 1];
        current_poly[0] = Complex64::new(1.0, 0.0); // Constant term
        current_poly[problem.weights[0] as usize] = Complex64::new(1.0, 0.0); // x^w_0 term

        // Multiply by (1 + x^w_i) for each remaining weight
        for &weight in &problem.weights[1..] {
            let mut new_poly = vec![Complex64::new(0.0, 0.0); current_poly.len() + weight as usize];
            
            // Copy current polynomial
            for (i, &coeff) in current_poly.iter().enumerate() {
                new_poly[i] = coeff;
            }
            
            // Add x^weight times current polynomial
            for (i, &coeff) in current_poly.iter().enumerate() {
                let new_index = i + weight as usize;
                if new_index < new_poly.len() {
                    new_poly[new_index] += coeff;
                }
            }
            
            current_poly = new_poly;
        }

        // Create formal power series from polynomial coefficients
        let series = FormalPowerSeries::new(current_poly);
        let idvbit = IDVBit::new_symbolic(series, "subset_sum_generating_function");
        
        self.idvbit_cache.insert(cache_key, idvbit.clone());
        Ok(idvbit)
    }

    /// Create IDVBit for Boolean SAT using characteristic function
    pub fn create_boolean_sat_idvbit(&mut self, problem: &SatProblem) -> IDVBitResult<IDVBit> {
        let cache_key = format!("sat_{}_{:?}", problem.variables, problem.clauses);
        
        if let Some(cached) = self.idvbit_cache.get(&cache_key) {
            return Ok(cached.clone());
        }

        if problem.variables > 16 {
            return Err(IDVBitError::ResourceLimit(
                format!("SAT instance too large: {} variables", problem.variables)
            ));
        }

        // Create characteristic generating function for satisfying assignments
        // Coefficient of x^k is 1 if assignment k satisfies formula, 0 otherwise
        use num_complex::Complex64;
        let total_assignments = 1usize << problem.variables;
        let mut coefficients = vec![Complex64::new(0.0, 0.0); total_assignments];
        
        for assignment in 0..total_assignments {
            if self.check_sat_assignment(problem, assignment as u64) {
                coefficients[assignment] = Complex64::new(1.0, 0.0);
            }
            // else coefficient remains 0
        }

        let series = FormalPowerSeries::new(coefficients);
        let idvbit = IDVBit::new_symbolic(series, "boolean_sat_characteristic_function");
        
        self.idvbit_cache.insert(cache_key, idvbit.clone());
        Ok(idvbit)
    }

    /// Extract solutions for subset sum problem
    pub fn extract_subset_sum_solutions(&mut self, problem: &SubsetSumProblem) -> IDVBitResult<Vec<Vec<usize>>> {
        let mut idvbit = self.create_subset_sum_idvbit(problem)?;
        let coefficient = self.extract_coefficient(&mut idvbit, problem.target as u64)?;
        
        // If coefficient is 0, no solutions
        if coefficient.real().is_zero() {
            return Ok(vec![]);
        }

        // For small instances, enumerate actual solutions
        if problem.weights.len() <= 10 {
            return Ok(self.enumerate_subset_solutions(problem));
        }

        // For larger instances, just return that solutions exist
        if !coefficient.real().is_zero() {
            Ok(vec![vec![]]) // Placeholder indicating solutions exist
        } else {
            Ok(vec![])
        }
    }

    /// Enumerate all subset sum solutions for small instances
    fn enumerate_subset_solutions(&self, problem: &SubsetSumProblem) -> Vec<Vec<usize>> {
        let mut solutions = Vec::new();
        let n = problem.weights.len();
        
        // Try all 2^n subsets
        for subset_bits in 0..(1 << n) {
            let mut sum = 0;
            let mut subset = Vec::new();
            
            for i in 0..n {
                if (subset_bits >> i) & 1 == 1 {
                    sum += problem.weights[i];
                    subset.push(i);
                }
            }
            
            if sum == problem.target {
                solutions.push(subset);
            }
        }
        
        solutions
    }

    /// Extract solutions for Boolean SAT problem
    pub fn extract_sat_solutions(&mut self, problem: &SatProblem) -> IDVBitResult<Vec<u64>> {
        let mut idvbit = self.create_boolean_sat_idvbit(problem)?;
        let mut solutions = Vec::new();
        
        let total_assignments = 1u64 << problem.variables;
        for assignment in 0..total_assignments {
            let coefficient = self.extract_coefficient(&mut idvbit, assignment)?;
            if !coefficient.real().is_zero() {
                solutions.push(assignment);
            }
        }
        
        Ok(solutions)
    }

    /// Extract coefficient range for analysis
    pub fn extract_coefficient_range(&mut self, idvbit: &mut IDVBit, start: u64, end: u64) -> IDVBitResult<Vec<Complex>> {
        let mut coefficients = Vec::new();
        
        for index in start..=end {
            let coeff = self.extract_coefficient(idvbit, index)?;
            coefficients.push(coeff);
        }
        
        Ok(coefficients)
    }

    /// Analyze coefficient growth for generating functions
    pub fn analyze_coefficient_growth(&mut self, idvbit: &mut IDVBit, max_index: u64) -> IDVBitResult<HashMap<String, f64>> {
        let coefficients = self.extract_coefficient_range(idvbit, 0, max_index)?;
        
        let mut analysis = HashMap::new();
        
        // Calculate basic statistics
        let real_parts: Vec<f64> = coefficients.iter()
            .map(|c| c.real().to_f64())
            .collect();
        
        let sum: f64 = real_parts.iter().sum();
        let mean = sum / real_parts.len() as f64;
        
        let variance: f64 = real_parts.iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f64>() / real_parts.len() as f64;
        
        let max_coeff = real_parts.iter().fold(0.0f64, |a, &b| a.max(b));
        let min_coeff = real_parts.iter().fold(0.0f64, |a, &b| a.min(b));
        
        analysis.insert("mean".to_string(), mean);
        analysis.insert("variance".to_string(), variance);
        analysis.insert("max".to_string(), max_coeff);
        analysis.insert("min".to_string(), min_coeff);
        analysis.insert("sum".to_string(), sum);
        
        Ok(analysis)
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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_subset_sum_extraction() {
        let mut extractor = IndexExtractor::new();
        let problem = SubsetSumProblem {
            weights: vec![1, 2, 3],
            target: 3,
        };
        
        let solutions = extractor.extract_subset_sum_solutions(&problem).unwrap();
        
        // Should find solutions: {3} and {1, 2}
        assert_eq!(solutions.len(), 2);
        
        // Verify solutions
        let mut found_single = false;
        let mut found_pair = false;
        
        for solution in solutions {
            let sum: u32 = solution.iter().map(|&i| problem.weights[i]).sum();
            assert_eq!(sum, problem.target);
            
            if solution.len() == 1 && solution[0] == 2 { // weight[2] = 3
                found_single = true;
            } else if solution.len() == 2 && solution.contains(&0) && solution.contains(&1) {
                found_pair = true; // weights[0] + weights[1] = 1 + 2 = 3
            }
        }
        
        assert!(found_single, "Should find single element solution containing 3");
        assert!(found_pair, "Should find pair solution containing 1+2");
    }

    #[test]
    fn test_subset_sum_no_solution() {
        let mut extractor = IndexExtractor::new();
        let problem = SubsetSumProblem {
            weights: vec![2, 4, 6],
            target: 5, // Impossible with even weights
        };
        
        let solutions = extractor.extract_subset_sum_solutions(&problem).unwrap();
        assert_eq!(solutions.len(), 0);
    }

    #[test]
    fn test_boolean_sat_extraction() {
        let mut extractor = IndexExtractor::new();
        let problem = SatProblem {
            variables: 2,
            clauses: vec![vec![1, 2]], // x1 OR x2
        };
        
        let solutions = extractor.extract_sat_solutions(&problem).unwrap();
        
        // Should have 3 satisfying assignments: 01, 10, 11 (binary)
        // Which correspond to assignments 1, 2, 3 (decimal)
        assert_eq!(solutions.len(), 3);
        assert!(solutions.contains(&1)); // x1=true, x2=false
        assert!(solutions.contains(&2)); // x1=false, x2=true  
        assert!(solutions.contains(&3)); // x1=true, x2=true
        assert!(!solutions.contains(&0)); // x1=false, x2=false (unsatisfying)
    }

    #[test]
    fn test_coefficient_extraction_caching() {
        let mut extractor = IndexExtractor::new();
        let problem = SubsetSumProblem {
            weights: vec![1, 2],
            target: 2,
        };
        
        let mut idvbit = extractor.create_subset_sum_idvbit(&problem).unwrap();
        
        // First extraction
        let coeff1 = extractor.extract_coefficient(&mut idvbit, 2).unwrap();
        
        // Second extraction should use cache
        let coeff2 = extractor.extract_coefficient(&mut idvbit, 2).unwrap();
        
        assert_eq!(coeff1.real().to_f64(), coeff2.real().to_f64());
        // Should be 1.0 for subset sum {2}
    }

    #[test]
    fn test_coefficient_growth_analysis() {
        let mut extractor = IndexExtractor::new();
        let problem = SubsetSumProblem {
            weights: vec![1, 1, 1], // All weight 1
            target: 3,
        };
        
        let mut idvbit = extractor.create_subset_sum_idvbit(&problem).unwrap();
        let analysis = extractor.analyze_coefficient_growth(&mut idvbit, 5).unwrap();
        
        assert!(analysis.contains_key("mean"));
        assert!(analysis.contains_key("max"));
        assert!(analysis.contains_key("sum"));
        
        // For weights [1,1,1], generating function is (1+x)^3 = 1 + 3x + 3x^2 + x^3
        let max_coeff = analysis.get("max").unwrap();
        assert_eq!(*max_coeff, 3.0); // Coefficient of x^1 and x^2
    }
}