# Contributing to Dimensional Mathematics

Thanks for your interest! This is a research project for exploring dimensional geometry and mathematics.

## Quick Start

1. **Fork and clone** the repository
2. **Set up environment**:

   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -e ".[dev]"
   ```

3. **Run tests and linter** to ensure 100% GREEN SMILES: `pytest`, `ruff check`
4. **Make your changes** and add new tests when productive
5. **Submit a pull request**

## What We Accept

- ✅ **Bug fixes** - Just fix it and add a test
- ✅ **New mathematical functions** - With tests showing they work
- ✅ **Performance improvements** - With benchmarks showing the improvement
- ✅ **Documentation improvements** - Clearer explanations, better examples

## What We Don't Accept

- ❌ **Breaking changes** to core functions (`v`, `s`, `c`, `r`)
- ❌ **Code without tests** - New code needs >80% test coverage
- ❌ **Performance regressions** - Don't make things slower

## Testing Mathematical Functions

When adding new mathematical functions:

- Include tests with known exact values (e.g., V(2) = π)
- Test mathematical properties (e.g., recurrence relations)
- Use property-based testing for relationships

## Code Style

- Functions should have docstrings with mathematical formulas
- Use `numpy` and `scipy` for numerical operations
- Keep mathematical core simple and clean
