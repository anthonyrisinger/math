# Comprehensive Test Suite Architecture

## 🎯 Testing Philosophy

**Mathematical Correctness First**: Every mathematical property, edge case, and numerical behavior must be verified. This is a mathematical library where correctness is non-negotiable.

## 🏗️ Test Architecture

### **Test Categories & Hierarchy**

```
tests/
├── conftest.py                    # Shared pytest configuration & fixtures
├── test_mathematical_properties/  # Core mathematical correctness
│   ├── test_gamma_properties.py   # Γ(n+1) = n!, Γ(1/2) = √π, etc.
│   ├── test_dimensional_relations.py # V×S relationships, peak locations
│   ├── test_phase_conservation.py # Energy conservation, invariants
│   └── test_morphic_identities.py # φ² = φ+1, ψ² = 1-ψ, etc.
│
├── test_numerical_stability/      # Edge cases & numerical robustness  
│   ├── test_large_values.py      # Behavior at extreme parameter values
│   ├── test_small_values.py      # Near-zero and epsilon-scale testing
│   ├── test_overflow_handling.py # Gamma function overflow protection
│   └── test_convergence.py       # Numerical convergence properties
│
├── test_unit/                    # Individual function testing
│   ├── test_dimensional/
│   ├── test_visualization/  
│   └── test_analysis/
│
├── test_integration/             # Component interaction testing
│   ├── test_api_workflows.py    # End-to-end user workflows
│   ├── test_module_interfaces.py # Module boundary interactions
│   └── test_data_consistency.py # Consistent results across modules
│
├── test_regression/              # Prevent functionality breaking
│   ├── test_known_results.py    # Results that must never change
│   ├── test_api_compatibility.py # API stability across versions
│   └── golden_reference/         # Reference data for comparisons
│
└── benchmarks/                   # Performance & scalability
    ├── benchmark_core_functions.py
    ├── benchmark_visualization.py
    └── performance_regression.py
```

## 🔬 Mathematical Property Tests

### **Core Gamma Function Properties**
```python
class TestGammaProperties:
    """Test fundamental gamma function mathematical properties."""
    
    @pytest.mark.parametrize("n", range(1, 10))
    def test_factorial_property(self, n):
        """Γ(n+1) = n! for positive integers"""
        assert abs(gamma_extended(n + 1) - math.factorial(n)) < 1e-12
    
    def test_half_integer_formula(self):
        """Γ(1/2) = √π exactly"""
        assert abs(gamma_extended(0.5) - math.sqrt(math.pi)) < 1e-15
    
    def test_reflection_formula(self):
        """Γ(z)Γ(1-z) = π/sin(πz) for non-integer z"""
        z = 0.3
        left = gamma_extended(z) * gamma_extended(1 - z)
        right = math.pi / math.sin(math.pi * z)
        assert abs(left - right) < 1e-12
    
    def test_duplication_formula(self):
        """Γ(z)Γ(z+1/2) = √π * 2^(1-2z) * Γ(2z)"""
        z = 0.7
        left = gamma_extended(z) * gamma_extended(z + 0.5)
        right = math.sqrt(math.pi) * (2**(1 - 2*z)) * gamma_extended(2*z)
        assert abs(left - right) < 1e-10
```

### **Dimensional Measure Properties**
```python
class TestDimensionalMeasureProperties:
    """Test mathematical relationships between dimensional measures."""
    
    def test_volume_surface_relationship(self):
        """V_d = ∫ S_{d-1} dr from 0 to 1"""
        # This tests the fundamental relationship between volume and surface
        for d in [1, 2, 3, 4, 5]:
            volume_calculated = ball_volume(d)
            surface_area = sphere_surface(d) 
            # V_d should equal surface integral
            expected_volume = surface_area / d  # Simplified relationship
            assert abs(volume_calculated - expected_volume) < 1e-10
    
    @pytest.mark.parametrize("d", [1, 2, 3, 4, 5])
    def test_scaling_properties(self, d):
        """Test scaling V_d(r) = r^d * V_d(1)"""
        r = 2.5
        scaled_volume = ball_volume(d, radius=r) 
        unit_volume = ball_volume(d, radius=1.0)
        expected = (r**d) * unit_volume
        assert abs(scaled_volume - expected) < 1e-12
    
    def test_complexity_peak_location(self):
        """Complexity measure C(d) = V(d)×S(d) peaks at d ≈ 6"""
        dimensions = np.linspace(5, 7, 1000)
        complexities = [complexity_measure(d) for d in dimensions]
        peak_idx = np.argmax(complexities)
        peak_dimension = dimensions[peak_idx]
        assert 5.5 < peak_dimension < 6.5, f"Peak at {peak_dimension}, expected ~6"
```

### **Phase Dynamics Conservation Laws**
```python
class TestPhaseConservation:
    """Test conservation laws in phase dynamics."""
    
    def test_total_energy_conservation(self):
        """Total phase energy must be conserved in isolated evolution."""
        engine = PhaseDynamicsEngine(max_dimensions=6)
        initial_energy = engine.total_energy()
        
        # Evolve for many steps
        for _ in range(1000):
            engine.step(dt=0.01)
            
        final_energy = engine.total_energy()
        energy_drift = abs(final_energy - initial_energy)
        
        # Energy conservation should be exact to numerical precision
        assert energy_drift < 1e-12, f"Energy drift: {energy_drift}"
    
    def test_topological_invariant_preservation(self):
        """Chern numbers and winding numbers must remain integer."""
        engine = PhaseDynamicsEngine(max_dimensions=6)
        
        # Inject energy to create interesting dynamics
        engine.inject_energy(dimension=2, amount=0.5)
        
        for step in range(100):
            engine.step(dt=0.05)
            
            # Check that all topological charges remain integer
            chern_numbers = engine.get_chern_numbers()
            for dim, chern in chern_numbers.items():
                assert abs(chern - round(chern)) < 1e-10, f"Non-integer Chern number at step {step}"
```

## ⚡ Property-Based Testing with Hypothesis

```python
from hypothesis import given, strategies as st
from hypothesis.extra.numpy import arrays

class TestPropertyBased:
    """Use property-based testing for mathematical laws."""
    
    @given(st.floats(min_value=0.1, max_value=10.0))
    def test_gamma_monotonicity(self, x):
        """Γ(x+1) > Γ(x) for x > 0 (gamma is log-convex)"""
        gamma_x = gamma_extended(x)
        gamma_x_plus_1 = gamma_extended(x + 1)
        assert gamma_x_plus_1 > gamma_x
    
    @given(st.floats(min_value=0.1, max_value=50.0))
    def test_dimensional_measures_positive(self, d):
        """All dimensional measures should be positive for d > 0"""
        volume = ball_volume(d)
        surface = sphere_surface(d) 
        complexity = complexity_measure(d)
        
        assert volume > 0, f"Volume negative at d={d}"
        assert surface > 0, f"Surface negative at d={d}" 
        assert complexity > 0, f"Complexity negative at d={d}"
        assert np.isfinite(volume), f"Volume infinite at d={d}"
    
    @given(arrays(np.float64, shape=(5,), elements=st.floats(0.0, 5.0)))
    def test_phase_density_normalization(self, initial_phases):
        """Phase density evolution preserves total probability."""
        # Normalize input
        initial_phases = initial_phases / np.sum(initial_phases)
        
        engine = PhaseDynamicsEngine(max_dimensions=5)
        engine.set_phase_density(initial_phases)
        
        # Evolve
        for _ in range(50):
            engine.step(0.01)
            
        final_phases = engine.get_phase_density()
        total_final = np.sum(np.abs(final_phases)**2)
        
        # Probability conservation
        assert abs(total_final - 1.0) < 1e-10
```

## 📊 Performance Benchmarks

```python
class TestPerformance:
    """Performance benchmarks with regression detection."""
    
    def test_gamma_function_speed(self, benchmark):
        """Benchmark gamma function calculation speed."""
        values = np.linspace(0.1, 100, 10000)
        
        def gamma_calculations():
            return [gamma_extended(v) for v in values]
        
        result = benchmark(gamma_calculations)
        # Should complete 10k calculations in reasonable time
        assert len(result) == 10000
    
    def test_phase_evolution_scaling(self):
        """Test that phase evolution scales reasonably with dimension count."""
        times = []
        
        for max_dim in [4, 8, 12, 16]:
            engine = PhaseDynamicsEngine(max_dimensions=max_dim)
            
            start = time.time()
            for _ in range(100):
                engine.step(0.01)
            elapsed = time.time() - start
            
            times.append((max_dim, elapsed))
        
        # Should scale sub-quadratically
        for i in range(1, len(times)):
            dim_ratio = times[i][0] / times[i-1][0]  
            time_ratio = times[i][1] / times[i-1][1]
            
            assert time_ratio < dim_ratio**2, f"Worse than quadratic scaling: {time_ratio} vs {dim_ratio}²"
```

## 🔧 Test Configuration

### **`conftest.py`**
```python
import pytest
import numpy as np
import warnings
from dimensional import *

# Configure numpy for testing
np.seterr(all='raise')  # Convert warnings to errors
warnings.filterwarnings('error')  # Mathematical warnings are errors

@pytest.fixture
def golden_engine():
    """Standard test engine with known good parameters."""
    return PhaseDynamicsEngine(max_dimensions=6, timestep=0.01)

@pytest.fixture  
def test_dimensions():
    """Standard set of test dimensions covering all regimes."""
    return [0, 0.5, 1, 1.5, 2, 3, 4, 5, 6, 7, 8, 10, 15]

@pytest.fixture
def mathematical_tolerance():
    """Standard tolerance for mathematical comparisons."""
    return 1e-12

@pytest.fixture
def known_gamma_values():
    """Dictionary of known exact gamma function values."""
    return {
        1: 1.0,
        2: 1.0, 
        3: 2.0,
        4: 6.0,
        0.5: math.sqrt(math.pi),
        1.5: math.sqrt(math.pi) / 2,
        2.5: (3 * math.sqrt(math.pi)) / 4
    }

# Mark slow tests
def pytest_configure(config):
    config.addinivalue_line("markers", "slow: marks tests as slow")
    config.addinivalue_line("markers", "integration: marks integration tests")
    config.addinivalue_line("markers", "mathematical: marks core mathematical property tests")

# Custom test collection
def pytest_collection_modifyitems(config, items):
    # Run mathematical property tests first
    math_tests = [item for item in items if "mathematical" in item.keywords]
    other_tests = [item for item in items if item not in math_tests]
    items[:] = math_tests + other_tests
```

### **`pytest.ini`**
```ini
[tool:pytest]
testpaths = tests
python_files = test_*.py *_test.py
python_classes = Test*
python_functions = test_*
addopts = 
    --strict-markers
    --strict-config  
    --verbose
    -ra
    --cov=dimensional
    --cov=visualization
    --cov=analysis
    --cov-report=term-missing
    --cov-report=html:htmlcov
    --cov-fail-under=90
markers =
    slow: marks tests as slow (deselect with '-m "not slow"')
    integration: marks tests as integration tests
    mathematical: marks core mathematical property tests
    benchmark: performance benchmark tests
filterwarnings =
    error
    ignore::UserWarning
    ignore::DeprecationWarning:matplotlib.*
```

## 🎯 Test Coverage Requirements

### **Minimum Coverage Thresholds**
- **Overall**: 90% line coverage
- **Core mathematical functions**: 95% line coverage  
- **Public APIs**: 100% line coverage
- **Critical paths**: 100% branch coverage

### **Test Categories Distribution**
- **Mathematical Properties**: ~40% of tests
- **Unit Tests**: ~30% of tests
- **Integration Tests**: ~15% of tests
- **Edge Cases/Stability**: ~10% of tests
- **Performance/Benchmarks**: ~5% of tests

## 🚀 Continuous Integration

### **Test Execution Strategy**
```yaml
# Fast feedback loop
- Unit tests (< 30 seconds)
- Mathematical property tests (< 2 minutes) 
- Basic integration tests (< 5 minutes)

# Comprehensive validation  
- Full test suite (< 15 minutes)
- Performance benchmarks (< 10 minutes)
- Memory usage tests
- Cross-platform compatibility
```

### **Quality Gates**
- ✅ All mathematical property tests pass
- ✅ 90%+ code coverage achieved
- ✅ No performance regressions detected
- ✅ All API compatibility maintained
- ✅ Documentation examples executable

---

This comprehensive test architecture ensures that your mathematical library maintains correctness, performance, and reliability throughout the reorganization and future development.