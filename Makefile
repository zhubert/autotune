.PHONY: install install-rocm test clean

# Install dependencies (CUDA/CPU)
install:
	uv sync --dev

# Install with ROCm support (AMD GPUs, Linux only)
install-rocm:
	uv sync --dev --extra rocm

# Run all tests
test:
	uv run pytest tests/ -v

# Clean cache and build artifacts
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache htmlcov .coverage
