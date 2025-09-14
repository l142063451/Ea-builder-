"""
Tests for God-Index Python Implementation

Comprehensive test suite validating the God-Index system for problem-to-solution
mapping using mathematical techniques from symbolic math and IDVBit foundations.
"""

import pytest
import numpy as np
from research.god_index import (
    GodIndexCore, ProblemMapper, IndexExtractor, Verifier,
    SubsetSumProblem, SatProblem, GraphColoringProblem,
    ProblemClass, GodIndexResult,
    create_god_index_for_subset_sum, create_god_index_for_boolean_sat, create_general_god_index
)

class TestProblemClassification:
    """Test problem classification functionality."""
    
    def test_problem_classification(self):
        mapper = ProblemMapper()
        
        assert mapper.classify_problem("CNF formula") == ProblemClass.BOOLEAN_SAT
        assert mapper.classify_problem("subset sum problem") == ProblemClass.SUBSET_SUM
        assert mapper.classify_problem("graph coloring") == ProblemClass.GRAPH_COLORING
        assert mapper.classify_problem("knapsack optimization") == ProblemClass.KNAPSACK
        assert mapper.classify_problem("general problem") == ProblemClass.COMBINATORIAL

class TestSubsetSumMapping:
    """Test subset sum problem mapping and solving."""
    
    def test_subset_sum_small(self):
        mapper = ProblemMapper()
        problem = SubsetSumProblem(weights=[1, 2, 3], target=4)
        
        result = mapper.map_subset_sum(problem)
        assert result.result_type == "parameterized"
        assert result.data["params"] == [1.0, 2.0, 3.0]
    
    def test_subset_sum_impossible(self):
        mapper = ProblemMapper()
        problem = SubsetSumProblem(weights=[1, 2], target=10)  # Impossible target
        
        result = mapper.map_subset_sum(problem)
        assert result.result_type == "direct_index"
        assert result.data["index"] == 0  # No solutions
    
    def test_subset_sum_large(self):
        mapper = ProblemMapper()
        problem = SubsetSumProblem(weights=list(range(1, 22)), target=100)  # 21 weights
        
        result = mapper.map_subset_sum(problem)
        assert result.result_type == "compilation_required"
        assert result.data["problem_class"] == "SubsetSum"
    
    def test_subset_sum_extraction(self):
        extractor = IndexExtractor()
        problem = SubsetSumProblem(weights=[1, 2, 3], target=3)
        
        solutions = extractor.extract_subset_sum_solutions(problem)
        
        # Should find solutions: {3} and {1, 2} 
        assert len(solutions) == 2
        
        # Verify solutions by checking indices
        found_single = False
        found_pair = False
        
        for solution in solutions:
            solution_sum = sum(problem.weights[i] for i in solution)
            assert solution_sum == problem.target
            
            if len(solution) == 1 and solution[0] == 2:  # weight[2] = 3
                found_single = True
            elif len(solution) == 2 and 0 in solution and 1 in solution:  # weight[0] + weight[1] = 1 + 2 = 3
                found_pair = True
        
        assert found_single, "Should find single element solution containing 3"
        assert found_pair, "Should find pair solution containing 1+2"
    
    def test_subset_sum_no_solution(self):
        extractor = IndexExtractor()
        problem = SubsetSumProblem(weights=[2, 4, 6], target=5)  # Impossible with even weights
        
        solutions = extractor.extract_subset_sum_solutions(problem)
        assert len(solutions) == 0

class TestBooleanSatMapping:
    """Test Boolean SAT problem mapping and solving."""
    
    def test_boolean_sat_small(self):
        mapper = ProblemMapper()
        problem = SatProblem(variables=2, clauses=[[1], [-2]])  # x1 AND NOT x2
        
        result = mapper.map_boolean_sat(problem)
        assert result.result_type == "direct_index"
        assert result.data["index"] == 1  # One satisfying assignment: x1=true, x2=false
    
    def test_sat_solution_counting(self):
        mapper = ProblemMapper()
        problem = SatProblem(variables=2, clauses=[[1, 2]])  # x1 OR x2
        
        count = mapper._count_sat_solutions(problem)
        assert count == 3  # Three satisfying assignments: (1,0), (0,1), (1,1)
    
    def test_boolean_sat_extraction(self):
        extractor = IndexExtractor()
        problem = SatProblem(variables=2, clauses=[[1, 2]])  # x1 OR x2
        
        solutions = extractor.extract_sat_solutions(problem)
        
        # Should have 3 satisfying assignments: 01, 10, 11 (binary)
        # Which correspond to assignments 1, 2, 3 (decimal)
        assert len(solutions) == 3
        assert 1 in solutions  # x1=true, x2=false
        assert 2 in solutions  # x1=false, x2=true  
        assert 3 in solutions  # x1=true, x2=true
        assert 0 not in solutions  # x1=false, x2=false (unsatisfying)
    
    def test_boolean_sat_large(self):
        mapper = ProblemMapper()
        problem = SatProblem(variables=20, clauses=[[1, 2, 3]])  # Large instance
        
        result = mapper.map_boolean_sat(problem)
        assert result.result_type == "compilation_required"
        assert result.data["problem_class"] == "BooleanSAT"

class TestGraphColoringMapping:
    """Test graph coloring problem mapping."""
    
    def test_graph_coloring_small(self):
        mapper = ProblemMapper()
        problem = GraphColoringProblem(
            vertices=3,
            edges=[(0, 1), (1, 2)],  # Path graph
            colors=2
        )
        
        result = mapper.map_graph_coloring(problem)
        assert result.result_type == "direct_index"
        assert result.data["index"] > 0  # Should have valid colorings
    
    def test_graph_coloring_large(self):
        mapper = ProblemMapper()
        problem = GraphColoringProblem(vertices=15, edges=[], colors=3)  # Large instance
        
        result = mapper.map_graph_coloring(problem)
        assert result.result_type == "compilation_required"
        assert result.data["problem_class"] == "GraphColoring"

class TestSolutionVerification:
    """Test solution verification functionality."""
    
    def test_subset_sum_verification_valid(self):
        verifier = Verifier()
        problem = SubsetSumProblem(weights=[1, 2, 3, 4], target=5)
        
        # Valid solution: indices 0 and 3 (weights 1 + 4 = 5)
        solution = [0, 3]  # weights[0] + weights[3] = 1 + 4 = 5
        assert verifier.verify_subset_sum(problem, solution)
        
        # Another valid solution: indices 1 and 2 (weights 2 + 3 = 5)
        solution2 = [1, 2]  # weights[1] + weights[2] = 2 + 3 = 5
        assert verifier.verify_subset_sum(problem, solution2)
    
    def test_subset_sum_verification_invalid(self):
        verifier = Verifier()
        problem = SubsetSumProblem(weights=[1, 2, 3, 4], target=5)
        
        # Invalid solution: wrong sum
        solution = [0, 1]  # weights[0] + weights[1] = 1 + 2 = 3 ≠ 5
        assert not verifier.verify_subset_sum(problem, solution)
        
        # Invalid solution: out of bounds index
        solution2 = [0, 10]  # index 10 doesn't exist
        assert not verifier.verify_subset_sum(problem, solution2)
    
    def test_boolean_sat_verification(self):
        verifier = Verifier()
        problem = SatProblem(
            variables=3,
            clauses=[[1, 2], [-1, 3], [-2, -3]]  # (x1 ∨ x2) ∧ (¬x1 ∨ x3) ∧ (¬x2 ∨ ¬x3)
        )
        
        # Test satisfying assignment: x1=true, x2=false, x3=true (binary 101 = decimal 5)
        assert verifier.verify_boolean_sat(problem, 5)
        
        # Test unsatisfying assignment: x1=true, x2=true, x3=true (binary 111 = decimal 7)
        # This violates clause 3: (¬x2 ∨ ¬x3) since both x2 and x3 are true
        assert not verifier.verify_boolean_sat(problem, 7)
    
    def test_graph_coloring_verification(self):
        verifier = Verifier()
        problem = GraphColoringProblem(
            vertices=3,
            edges=[(0, 1), (1, 2)],  # Path graph: 0-1-2
            colors=2
        )
        
        # Valid coloring: alternating colors
        coloring = [0, 1, 0]  # vertex 0: color 0, vertex 1: color 1, vertex 2: color 0
        assert verifier.verify_graph_coloring(problem, coloring)
        
        # Invalid coloring: adjacent vertices same color
        invalid_coloring = [0, 0, 1]  # vertices 0 and 1 both have color 0
        assert not verifier.verify_graph_coloring(problem, invalid_coloring)
        
        # Invalid coloring: out of range color
        invalid_coloring2 = [0, 1, 3]  # color 3 doesn't exist (only colors 0,1)
        assert not verifier.verify_graph_coloring(problem, invalid_coloring2)
    
    def test_verification_report(self):
        verifier = Verifier()
        results = [True, True, False, True, False]
        report = verifier.generate_verification_report(results)
        
        assert report["total_verifications"] == 5.0
        assert report["valid_solutions"] == 3.0
        assert report["invalid_solutions"] == 2.0
        assert report["success_rate"] == 0.6

class TestGodIndexCore:
    """Test the complete God-Index system integration."""
    
    def test_subset_sum_end_to_end(self):
        core = GodIndexCore()
        problem = SubsetSumProblem(weights=[1, 2, 3, 4], target=5)
        
        solutions = core.solve_subset_sum(problem)
        
        # Should find valid solutions
        assert len(solutions) > 0, "Should find at least one solution"
        
        # Verify each solution manually
        for solution in solutions:
            solution_sum = sum(problem.weights[i] for i in solution)
            assert solution_sum == problem.target, "Solution should sum to target"
    
    def test_boolean_sat_end_to_end(self):
        core = GodIndexCore()
        problem = SatProblem(variables=2, clauses=[[1, 2]])  # x1 OR x2
        
        assignments = core.solve_boolean_sat(problem)
        
        # Should find 3 satisfying assignments
        assert len(assignments) == 3
        assert 1 in assignments  # 01: x1=true, x2=false
        assert 2 in assignments  # 10: x1=false, x2=true
        assert 3 in assignments  # 11: x1=true, x2=true
    
    def test_performance_stats(self):
        core = GodIndexCore()
        
        # Solve a problem to populate caches
        problem = SubsetSumProblem(weights=[1, 2], target=3)
        _ = core.solve_subset_sum(problem)
        
        stats = core.get_performance_stats()
        assert "cached_solutions" in stats
        assert stats["cached_solutions"] > 0
    
    def test_complexity_analysis(self):
        core = GodIndexCore()
        
        analysis = core.analyze_complexity(ProblemClass.SUBSET_SUM, [10, 15])
        assert analysis["problem_class"] == "subset_sum"
        assert analysis["complexity"] == "Tractable"
        
        large_analysis = core.analyze_complexity(ProblemClass.SUBSET_SUM, [30, 100])
        assert large_analysis["complexity"] == "Large"
    
    def test_caching(self):
        core = GodIndexCore()
        problem = SubsetSumProblem(weights=[1, 3], target=4)
        
        # First solve
        solutions1 = core.solve_subset_sum(problem)
        
        # Second solve should use cache
        solutions2 = core.solve_subset_sum(problem)
        
        assert len(solutions1) == len(solutions2)
        
        # Clear cache and verify
        core.clear_caches()
        stats = core.get_performance_stats()
        assert stats["cached_solutions"] == 0

class TestFactoryFunctions:
    """Test factory functions for creating specialized God-Index instances."""
    
    def test_factory_methods(self):
        subset_core = create_god_index_for_subset_sum()
        sat_core = create_god_index_for_boolean_sat()
        general_core = create_general_god_index()
        
        # All should be functional GodIndexCore instances
        assert isinstance(subset_core, GodIndexCore)
        assert isinstance(sat_core, GodIndexCore)
        assert isinstance(general_core, GodIndexCore)

class TestCaching:
    """Test caching functionality across components."""
    
    def test_problem_mapper_caching(self):
        mapper = ProblemMapper()
        problem = SubsetSumProblem(weights=[1, 2], target=3)
        
        # First call
        result1 = mapper.map_subset_sum(problem)
        
        # Second call should use cache
        result2 = mapper.map_subset_sum(problem)
        
        assert result1.result_type == result2.result_type
        assert result1.data == result2.data
    
    def test_index_extractor_caching(self):
        extractor = IndexExtractor()
        problem = SubsetSumProblem(weights=[1, 2], target=2)
        
        # Create IDVBit twice - second should use cache
        idvbit1 = extractor.create_subset_sum_idvbit(problem)
        idvbit2 = extractor.create_subset_sum_idvbit(problem)
        
        # Should be the same object from cache
        assert idvbit1 is idvbit2
    
    def test_verifier_caching(self):
        verifier = Verifier()
        problem = SubsetSumProblem(weights=[1, 2, 3], target=3)
        solution = [2]  # weight[2] = 3
        
        # First verification
        result1 = verifier.verify_subset_sum(problem, solution)
        
        # Second verification should use cache
        result2 = verifier.verify_subset_sum(problem, solution)
        
        assert result1 == result2
        assert result1  # Should be valid

class TestMathematicalProperties:
    """Test mathematical properties and correctness."""
    
    def test_subset_sum_generating_function(self):
        extractor = IndexExtractor()
        
        # Test with weights [1, 1, 1], generating function should be (1+x)^3 = 1 + 3x + 3x^2 + x^3
        problem = SubsetSumProblem(weights=[1, 1, 1], target=2)
        idvbit = extractor.create_subset_sum_idvbit(problem)
        
        # Coefficient of x^2 should be 3 (three ways to get sum 2: choose any 2 of the 3 weights)
        coeff = idvbit.query_coefficient(2)
        assert abs(coeff - 3.0) < 1e-10, f"Expected coefficient 3.0, got {coeff}"
    
    def test_coefficient_growth_analysis(self):
        extractor = IndexExtractor()
        problem = SubsetSumProblem(weights=[1, 1, 1], target=3)
        
        idvbit = extractor.create_subset_sum_idvbit(problem)
        analysis = extractor.analyze_coefficient_growth(idvbit, 5)
        
        assert "mean" in analysis
        assert "max" in analysis
        assert "sum" in analysis
        
        # For weights [1,1,1], generating function is (1+x)^3 = 1 + 3x + 3x^2 + x^3
        # So max coefficient should be 3
        assert analysis["max"] == 3.0
    
    def test_sat_characteristic_function(self):
        extractor = IndexExtractor()
        
        # Simple SAT: x1 (only x1=true satisfies)
        problem = SatProblem(variables=1, clauses=[[1]])
        idvbit = extractor.create_boolean_sat_idvbit(problem)
        
        # Only assignment 1 (x1=true) should have coefficient 1
        assert abs(idvbit.query_coefficient(0)) < 1e-10  # x1=false
        assert abs(idvbit.query_coefficient(1) - 1.0) < 1e-10  # x1=true

class TestErrorHandling:
    """Test error handling and edge cases."""
    
    def test_large_sat_instance_error(self):
        extractor = IndexExtractor()
        problem = SatProblem(variables=20, clauses=[[1]])  # Too large
        
        with pytest.raises(ValueError, match="SAT instance too large"):
            extractor.create_boolean_sat_idvbit(problem)
    
    def test_empty_weights_subset_sum(self):
        # Test edge case with no weights - should find empty solution for target 0
        problem = SubsetSumProblem(weights=[], target=0)
        
        extractor = IndexExtractor()
        solutions = extractor.extract_subset_sum_solutions(problem)
        
        # For empty weights and target 0, the empty subset sums to 0 - this should be valid
        # However, our generating function will be 1 (no weights), coefficient of x^0 = 1
        # So we should find the empty solution
        assert len(solutions) >= 1, "Should find empty solution for target 0 with no weights"
    
    def test_invalid_graph_coloring_verification(self):
        verifier = Verifier()
        problem = GraphColoringProblem(vertices=2, edges=[(0, 1)], colors=2)
        
        # Test with wrong number of vertices in coloring
        assert not verifier.verify_graph_coloring(problem, [0])  # Too few colors
        assert not verifier.verify_graph_coloring(problem, [0, 1, 2])  # Too many colors