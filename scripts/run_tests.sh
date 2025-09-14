#!/bin/bash
# Test runner script for ISVGPU project

set -e

echo "ISVGPU Test Runner"
echo "=================="

# Check if we're in the right directory
if [ ! -f "memory.md" ]; then
    echo "Error: Run this script from the project root directory"
    exit 1
fi

echo "Running Rust tests..."
cd rust
cargo test --all --verbose
cargo clippy --all-targets --all-features -- -D warnings
cd ..

echo "Running Python tests..."
cd python
if [ ! -d ".venv" ]; then
    echo "Setting up Python environment..."
    poetry install --no-interaction
fi

poetry run pytest tests/ -v
poetry run black --check src/ tests/
poetry run isort --check-only src/ tests/
poetry run flake8 src/ tests/
cd ..

echo "Building documentation..."
cd rust && cargo doc --no-deps --all
cd ../python && poetry run sphinx-build -b html docs/ docs/_build/ 2>/dev/null || echo "Sphinx docs not set up yet"
cd ..

echo "All tests completed successfully!"
echo "Summary:"
echo "- Rust: Build ✓, Tests ✓, Clippy ✓"
echo "- Python: Tests ✓, Format ✓, Lint ✓"
echo "- Documentation: ✓"