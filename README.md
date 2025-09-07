# 🌌 Dimensional Mathematics Framework

<div align="center">

![Version](https://img.shields.io/badge/version-3.0.0-blue.svg)
![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![Tests](https://img.shields.io/badge/tests-163%20passing-brightgreen)
![License](https://img.shields.io/badge/license-MIT-green)

**Explore the beauty of higher-dimensional geometry through elegant mathematics and stunning visualizations.**

[Installation](#installation) • [Quick Start](#quick-start) • [Features](#features) • [API](#api) • [Gallery](#gallery)

</div>

---

## ✨ What is Dimensional?

Dimensional is a Python library for exploring n-dimensional geometry, gamma functions, and their fascinating relationships. It answers questions like:

- 📊 What's the volume of a 5-dimensional sphere?
- 🌐 At what dimension does surface area peak?
- 🎯 How do geometric measures behave as dimensions increase?

## 🚀 Installation

```bash
pip install -e .
```

## 🎨 Quick Start

### One-Line Magic

```python
import dimensional

# Explore dimension 4
dimensional.explore(4)
```

Output:
```
========================================
  Dimension d = 4
========================================
  Volume:     4.934802
  Surface:    19.739209
  Complexity: 97.409091
  Ratio:      4.000000
  Gamma:      6.000000
========================================
```

### Generate Beautiful Reports

```python
import dimensional

# Generate an HTML report with interactive plots
dimensional.generate_report(d=5.257, filename="my_analysis.html")
```

This creates a stunning HTML report with:
- 📈 Interactive Plotly visualizations
- 🎨 Beautiful gradient design
- 📊 Key metrics and insights
- ✨ Professional presentation

## 🔬 Core Features

### Mathematical Functions

```python
import dimensional as dm

# Core measures
dm.v(4)      # Volume of 4D unit ball: 4.934802
dm.s(4)      # Surface area of 4D sphere: 19.739209
dm.c(4)      # Complexity measure: 97.409091
dm.r(4)      # Ratio S/V: 4.0

# Gamma functions
dm.gamma(3.5)    # Γ(3.5) = 3.323351
dm.digamma(2)    # ψ(2) = 0.422784
```

### Geometric Objects

```python
from dimensional import Ball, Sphere

# Create geometric objects
ball = Ball(dimension=5, radius=2)
print(f"Volume: {ball.volume}")
print(f"Surface Area: {ball.surface_area}")

sphere = Sphere(dimension=3, radius=1)
print(f"Surface Area: {sphere.surface_area}")
print(f"Curvature: {sphere.curvature}")
```

### Visual Exploration

```python
import dimensional

# Find interesting dimensions
peaks = dimensional.peaks()
print(f"Volume peaks at d={peaks['volume'][0]:.3f}")
print(f"Surface peaks at d={peaks['surface'][0]:.3f}")

# Interactive laboratory
dimensional.lab()  # Launch interactive environment

# Instant visualization
dimensional.instant()  # Quick multi-panel plot
```

## 📊 Mathematical Foundation

The library implements these fundamental formulas:

- **Volume of n-ball**: `V(d) = π^(d/2) / Γ(d/2 + 1)`
- **Surface of n-sphere**: `S(d) = 2π^(d/2) / Γ(d/2)`
- **Complexity**: `C(d) = V(d) × S(d)`
- **Ratio**: `R(d) = S(d) / V(d)`

## 🎯 Key Insights

Our research reveals fascinating patterns:

- 📍 **Volume peaks** at dimension ≈ 5.257
- 📍 **Surface peaks** at dimension ≈ 7.257
- 📍 **Complexity peaks** at dimension ≈ 6.335
- 📉 **High dimensions** → Both volume and surface approach zero!

## 🛠️ Advanced Usage

### Batch Operations

```python
import numpy as np
import dimensional as dm

# Analyze multiple dimensions at once
dimensions = np.linspace(1, 10, 100)
volumes = dm.v(dimensions)
surfaces = dm.s(dimensions)
```

### HTML Report Generation

```python
# Generate comprehensive analysis report
dm.generate_report(d=None)  # Auto-finds optimal dimension
```

## 🏗️ Architecture

```
dimensional/
├── core.py          # Mathematical functions (148 lines)
├── geometry/        # Ball and Sphere classes
├── viz.py           # Visualization functions
├── explore.py       # Rich interactive features
└── report.py        # HTML report generator
```

**Clean. Focused. Beautiful.**

## 🧪 Testing

```bash
# Run all tests
pytest

# Current status: 163 tests passing (100% of active tests)
```

## 📈 Performance

- ⚡ **Fast imports** - No heavy dependencies
- 💾 **Small footprint** - Only 3,777 lines of code
- 🎯 **Focused API** - 32 carefully chosen functions
- 🚀 **NumPy optimized** - Vectorized operations

## 🎨 Gallery

### Dimension 4 - Our Universe
```python
dimensional.explore(4)  # Explore our 4D spacetime
```

### Golden Ratio Dimension
```python
dimensional.explore(1.618)  # φ, the golden ratio
```

### Peak Complexity
```python
dimensional.explore(6.335)  # Maximum complexity dimension
```

## 🤝 Contributing

We keep it simple:
1. Core math must have tests
2. Visual features must be beautiful
3. Code must be readable

## 📜 Version History

- **v3.0.0** - Complete rewrite. 76% less code, 100% more focus
- **v2.x** - Feature-rich research framework (deprecated)
- **v1.x** - Initial exploration

## 💫 Philosophy

> "Simplicity is the ultimate sophistication." - Leonardo da Vinci

We chose to make this library do one thing brilliantly: explore dimensional mathematics with beauty and clarity.

## 📝 License

MIT License - Use freely, create beauty.

---

<div align="center">

**Built with ❤️ by mathematicians who believe in beautiful code.**

[Report Issues](https://github.com/user/dimensional/issues) • [Documentation](https://dimensional.readthedocs.io) • [Examples](https://github.com/user/dimensional/examples)

</div>