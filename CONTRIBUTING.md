# ISVGPU Contributing Guidelines

## Welcome to ISVGPU Research Project

Thank you for your interest in contributing to the ISVGPU (Infinite Superposition Virtual GPU) research project. This project explores advanced mathematical techniques for computational acceleration while maintaining rigorous scientific standards.

## Code of Conduct

This project follows a professional and inclusive code of conduct. Please:
- Be respectful and constructive in all interactions
- Focus on technical merit and scientific rigor  
- Clearly distinguish proven techniques from speculative research
- Provide proper citations and references for mathematical claims

## Scientific Standards

### Content Classification
All contributions must clearly mark content as:
- **PROVEN**: Mathematically established techniques with proofs or strong evidence
- **HEURISTIC**: Practical approximations with empirical validation
- **SPECULATIVE**: Theoretical models or open research questions

### Documentation Requirements
- Update `memory.md` for all research contributions
- Include mathematical proofs or proof sketches where applicable
- Provide reproducible examples and test cases
- Document computational complexity and resource requirements

## Development Workflow

### Repository Structure
```
rust/           # Rust workspace with core libraries
python/         # Python orchestration and research tools
tests/          # Comprehensive test suites
docs/           # Documentation and mathematical derivations
```

### Pull Request Process
1. **Research Issues**: Create issue with `research` label for mathematical questions
2. **Implementation**: Follow the PR sequence: PR-001 through PR-011+
3. **Testing**: Include unit tests, integration tests, and mathematical validation
4. **Memory Update**: Always update `memory.md` with research notes and decisions
5. **Review**: Code review focuses on correctness, performance, and scientific rigor

### PR Naming Convention
- `feat/pr-XXX/description` for feature branches
- `fix/issue-XXX/description` for bug fixes  
- `research/topic-name` for research investigations

## Technical Requirements

### Rust Code
- Follow `rustfmt` formatting
- Pass `clippy` linting with no warnings
- Include comprehensive documentation
- Benchmark performance-critical code
- Use `cargo test` for unit testing

### Python Code
- Use `black` for formatting
- Use `isort` for import organization
- Follow `flake8` linting standards
- Include type hints where applicable
- Use `pytest` for testing

### Mathematical Code
- Implement exact arithmetic where possible
- Validate numerical stability
- Document algorithm complexity
- Provide reference implementations
- Include convergence analysis

## Contribution Types

### Mathematical Research
- Formal proofs and derivations
- Algorithm analysis and optimization
- Complexity theoretical results
- Numerical methods validation

### Implementation
- Core mathematical libraries
- Performance optimization
- API design and compatibility
- Testing and validation

### Documentation
- Mathematical exposition
- User guides and tutorials
- API documentation
- Research notes and findings

### Testing
- Unit tests for mathematical functions
- Integration tests for system components
- Performance benchmarks
- Correctness validation

## Review Criteria

### Code Quality
- Correctness and robustness
- Performance and efficiency
- Maintainability and clarity
- Documentation completeness

### Scientific Rigor
- Mathematical accuracy
- Proper classification (PROVEN/HEURISTIC/SPECULATIVE)
- Citation of sources
- Reproducibility

### Project Alignment
- Contribution to project goals
- Integration with existing components
- Memory.md updates
- Test coverage

## Getting Started

1. **Read the documentation**: Start with README.md and memory.md
2. **Set up environment**: Install Rust, Python, and dependencies
3. **Run tests**: Ensure your setup works correctly
4. **Pick an issue**: Look for `good-first-issue` or `research` labels
5. **Create PR**: Follow the development workflow

## Questions and Support

- **General questions**: Create issue with `question` label
- **Research discussions**: Use `research` label and provide context
- **Implementation help**: Tag issues with relevant component labels
- **Bug reports**: Include reproduction steps and environment details

## License and Legal

- All contributions licensed under MIT License
- Research contributions may be published with proper attribution
- No cryptographic attacks on production systems
- No kernel security bypasses

---

**Remember**: This is a research project exploring theoretical computational models. Maintain scientific integrity and clearly distinguish between proven capabilities and speculative ideas.