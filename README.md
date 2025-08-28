# Dimensional Mathematics Framework  
## Unified Mathematical Package with Modern Architecture

[![Tests](https://img.shields.io/badge/tests-263%20passed%2C%2012%20failed-yellow)]()
[![Status](https://img.shields.io/badge/status-active%20development-orange)]()

**Mathematical framework for dimensional analysis, gamma functions, and numerical computation. Core mathematical functions are stable; some advanced features under development.**

> ⚠️ **Status**: Core mathematical functions (gamma, dimensional measures) are working and tested. Advanced features (enhanced_lab, parameter sweeps, some visualizations) may be non-functional or incomplete.

## 🚀 Quick Start

```bash
# Install the framework
pip install -e .

# Test basic functionality
python -c "import dimensional; print(dimensional.v(4.0))"  # Should output: 4.934802...

# Basic mathematical functions (VERIFIED WORKING)
python -c "import dimensional as d; print(f'V(4)={d.v(4):.3f}, S(4)={d.s(4):.3f}')"

# CLI interface (basic functionality)
python -m dimensional --help

# NOTE: Advanced features (enhanced_lab, complex visualizations) may not work
```

---

## 🏗️ Architecture

The framework is built with modern Python practices and provides both programmatic APIs and command-line tools:

```
dimensional/
├── dimensional/          # Core mathematical library
│   ├── __init__.py      # Main API exports
│   ├── cli.py           # Full-featured CLI with Typer + Rich
│   ├── gamma.py         # Gamma functions with safety & visualization
│   ├── measures.py      # Dimensional measures (V, S, C)
│   ├── phase.py         # Phase dynamics engine
│   ├── morphic.py       # Golden ratio mathematics
│   └── __main__.py      # CLI entry point
├── tests/               # Test suite (275 total: 263 pass, 12 fail)
│   ├── test_*.py        # Individual module tests
│   └── conftest.py      # Pytest configuration
├── analysis/            # Advanced analysis tools
├── core/                # Legacy mathematical core
└── scripts/             # Utility scripts
```

## 🎯 Key Features

### 🔬 Core Mathematical Functions (WORKING)
- **Numerically stable** gamma function implementations
- **Dimensional measures**: Volume V(d), Surface S(d), Complexity C(d)
- **Edge case handling** for mathematical edge cases
- **Validated against known mathematical properties**

### ⚡ Command Line Interface (BASIC)
- **Basic CLI** with help system
- **Mathematical computation** commands
- **Import/export** functionality for core features

### 🧪 Test Coverage (MIXED)
- **263 passing tests** for core mathematical functions
- **12 failing tests** in advanced features (convergence, threading, parameter sweeps)
- **Property-based testing** for mathematical correctness of core functions

### 🎨 Visualization (INCOMPLETE)
- **Some plotting capabilities** may be available
- **Advanced interactive features** likely non-functional
- **Export capabilities** status unknown

---

## 📱 Command Line Interface

The CLI provides access to basic mathematical functionality:

### Verified Working Commands

```bash
# Check CLI is available (WORKS)
python -m dimensional --help

# Basic mathematical computations (VERIFIED WORKING)
python -m dimensional v 4      # Volume: V(4.0) = 4.934802
python -m dimensional s 4      # Surface: S(4.0) = 19.739209  
python -m dimensional g 3.5    # Gamma: Γ(3.5) = 3.323351
```

### Status of Advanced Commands

> ⚠️ **Warning**: The following commands are listed in documentation but may not work:
> - `dimensional lab` (enhanced lab features)
> - `dimensional visualize` (complex visualization)
> - `dimensional demo` (comprehensive demonstration)
> - Parameter sweep functionality
> 
> Always test commands before relying on them.

---

## 🐍 Python API

### Verified Working Functions

```python
import dimensional as dm

# Core dimensional functions (TESTED AND WORKING)
volume = dm.v(4.0)       # 4D ball volume ≈ 4.935
surface = dm.s(4.0)      # 4D sphere surface ≈ 19.739  
complexity = dm.c(4.0)   # Combined complexity measure ≈ 97.41

# Gamma function family (numerically stable)
gamma_val = dm.gamma_safe(3.5)  # Standard gamma function with safety

# Basic constants and utilities
print(dm.PHI)           # Golden ratio
print(dm.PI)            # Pi constant
```

### Functions with Partial/Unknown Status

```python
# INTERACTIVE FEATURES (work but may have limitations):
dm.enhanced_lab(4.0)            # Enhanced interactive lab (launches successfully)

# UNKNOWN STATUS (verify before using):
# dm.peaks()                    # Find all peaks automatically  
# dm.explore()                  # Basic explore function
# dm.instant()                  # Generate visualizations
# dm.PhaseDynamicsEngine()      # Phase dynamics (has test failures)
```

---

## 🎨 Visualization Status

> ⚠️ **Status Unknown**: The codebase contains visualization modules but their functional status is unverified:
> - Plotly backend may or may not work
> - Kingdon Geometric Algebra features unverified
> - Export capabilities status unknown
> 
> Test visualization features before depending on them.

---

## 🧪 Testing & Quality

```bash
# Run all tests
pytest

# Current test status
pytest --tb=short  # Shows 275 tests: 263 passed, 12 failed
```

**Current Status**: 
- ✅ **263 tests passing** (core mathematical functions stable)
- ❌ **12 tests failing** (advanced features: convergence analysis, parameter sweeps, threading simulation)
- ⚠️ **4 tests skipped** (advanced features not implemented)

---

## 📦 Installation & Requirements

```bash
# Install from source
git clone https://github.com/user/dimensional-math
cd dimensional-math
pip install -e .

# Or install specific visualization backends
pip install -e ".[viz]"     # Include advanced visualization
pip install -e ".[dev]"     # Include development dependencies  
```

**Requirements**:
- Python 3.9+
- NumPy, SciPy (mathematical computing)
- Plotly (interactive visualization)
- Typer, Rich (modern CLI)
- Pydantic (type safety)
- Pytest (testing)

---

## 📊 Mathematical Foundation

This package provides numerical implementations of dimensional measures based on well-established mathematical formulas:

- **Volume of n-ball**: V(d) = π^(d/2) / Γ(d/2 + 1)
- **Surface of n-sphere**: S(d) = 2π^(d/2) / Γ(d/2)  
- **Complexity measure**: C(d) = V(d) × S(d)

The gamma function implementations use numerical stability techniques for reliable computation across the domain.

---

*This package provides computational tools for exploring dimensional measures and gamma function behavior. The mathematical relationships and patterns in dimensional space are interesting subjects for mathematical exploration and computational investigation.*