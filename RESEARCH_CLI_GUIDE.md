# Enhanced Interactive Research CLI Guide

## Dimensional Mathematics Research System

The Enhanced Interactive Research CLI provides powerful tools for discovering "the true shape of one" through dimensional mathematics research. This comprehensive system builds on the solid dimensional mathematics foundation with advanced research capabilities.

## 🚀 Quick Start

```bash
# Launch comprehensive research mode
python -m dimensional.cli research

# Run instant analysis
python -m dimensional.cli instant --config research

# Explore specific dimension with guided discovery
python -m dimensional.cli explore 5.26 --context peaks

# Run parameter sweep with export
python -m dimensional.cli sweep 4 6 --steps 50 --export
```

## 🔬 Core Research Functions

### Enhanced Lab (`lab()`)
Interactive research laboratory with session persistence, Rich visualization, and comprehensive analysis tools.

```python
from dimensional import enhanced_lab

# Launch with default settings
session = enhanced_lab()

# Load existing session 
session = enhanced_lab(session_id="lab_1234567890")

# Start at specific dimension
session = enhanced_lab(start_dimension=5.26)
```

**Interactive Commands:**
- `explore <dim>` - Analyze dimension in detail with Rich visualization
- `sweep <start> <end> [steps]` - Run parameter sweep across range
- `bookmark <name>` - Save current dimension for later reference
- `export` - Export data for publication-quality figures
- `save` - Save session to disk
- `load <id>` - Load previous session
- `critical` - Show critical dimensions tree
- `overview` - Display session summary

### Enhanced Explore (`explore()`)
Guided dimensional discovery with analysis paths and mathematical insights.

```python
from dimensional import enhanced_explore

# Basic exploration
results = enhanced_explore(4.0)

# Context-aware exploration
results = enhanced_explore(5.26, context="peaks")
results = enhanced_explore(3.14, context="critical") 
results = enhanced_explore(6.33, context="research")
```

**Discovery Paths:**
- **Peak Proximity**: Analyze distance to volume/surface/complexity peaks
- **Ratio Analysis**: Find mathematical constants in dimensional ratios
- **Derivative Analysis**: Identify critical points and rate changes
- **Local Neighborhoods**: Explore variance and behavior patterns
- **Critical Dimensions**: Proximity to known mathematical constants

### Enhanced Instant (`instant()`)
Multi-panel analysis with configurable research views.

```python
from dimensional import enhanced_instant

# Research configuration (default)
results = enhanced_instant("research")

# Peak-focused analysis
results = enhanced_instant("peaks")

# Discovery mode for pattern recognition
results = enhanced_instant("discovery")

# Publication-ready analysis
results = enhanced_instant("publication")
```

## 📊 Parameter Sweeps

### Interactive Parameter Sweeps
Real-time dimensional analysis across parameter ranges with Rich progress visualization.

```python
from dimensional.research_cli import InteractiveParameterSweep, RichVisualizer

visualizer = RichVisualizer()
sweeper = InteractiveParameterSweep(visualizer)

# Run dimensional sweep
sweep = sweeper.run_dimension_sweep(
    start=2.0, 
    end=8.0, 
    steps=100,
    notes="Research sweep around peaks"
)

# Display results with Rich tables
visualizer.show_parameter_sweep_analysis(sweep)
```

### CLI Parameter Sweeps
```bash
# Basic sweep
python -m dimensional.cli sweep 2 8 --steps 100

# With export to CSV
python -m dimensional.cli sweep 4 6 --steps 50 --export

# High-resolution sweep
python -m dimensional.cli sweep 5.2 5.3 --steps 1000 --export
```

## 💾 Session Management

### Research Session Persistence
Save and restore complete research sessions including data points, parameter sweeps, bookmarks, and notes.

```python
from dimensional.research_cli import ResearchPersistence, ResearchSession

# Initialize persistence system
persistence = ResearchPersistence()

# Save session
session_path = persistence.save_session(my_session)

# Load session
loaded_session = persistence.load_session("lab_1234567890")

# List available sessions
sessions = persistence.list_sessions()
for session_id, start_time in sessions:
    print(f"Session: {session_id} from {start_time}")
```

### CLI Session Management
```bash
# List all research sessions
python -m dimensional.cli sessions

# Launch lab with specific session
python -m dimensional.cli lab --session lab_1234567890
```

## 📈 Rich Terminal Visualization

### Mathematical Displays
Beautiful terminal visualizations using Rich formatting with mathematical precision.

```python
from dimensional.research_cli import RichVisualizer, ResearchPoint

visualizer = RichVisualizer()

# Create research point
point = ResearchPoint(
    dimension=5.26,
    volume=5.277766,
    surface=27.761050,
    complexity=146.516325,
    timestamp=datetime.now(),
    notes="Volume peak analysis"
)

# Rich dimensional analysis display
visualizer.show_dimensional_analysis(point)

# Critical dimensions tree
visualizer.show_critical_dimensions_tree()

# Session overview
visualizer.show_session_overview(session)
```

### Critical Dimensions Tree
Hierarchical display of mathematical constants and peak dimensions:

```
🌟 Critical Dimensions
├── 📐 Mathematical Constants
│   ├── π = 3.14159265
│   ├── e = 2.71828183
│   └── φ = 1.61803399 (Golden ratio)
├── 🏔️ Peak Dimensions
│   ├── Volume peak: d = 5.25643129, V = 5.27776797
│   ├── Surface peak: d = 7.25641128, S = 33.16119411
│   └── Complexity peak: d = 6.33540708, C = 161.70841158
└── 🔥 Critical Dimensions
    ├── volume_peak: d = 5.25643129
    ├── surface_peak: d = 7.25641128
    └── complexity_peak: d = 6.33540708
```

## 🎯 Publication-Quality Exports

### Export System
Comprehensive data export for publication-quality figures and analysis.

```python
from dimensional.research_cli import PublicationExporter

exporter = PublicationExporter()

# Export sweep data as CSV
csv_path = exporter.export_csv_data(parameter_sweep)

# Export complete session analysis as JSON
json_path = exporter.export_json_analysis(research_session)
```

### Export Formats

**CSV Data Export:**
```csv
dimension,volume,surface,complexity,timestamp,notes
4.0000,4.934802,19.739209,97.409091,2025-08-27T13:09:28,CLI parameter sweep
4.2222,5.045869,21.304779,107.501120,2025-08-27T13:09:28,CLI parameter sweep
```

**JSON Analysis Export:**
```json
{
  "session": {
    "session_id": "lab_1234567890",
    "start_time": "2025-08-27T13:09:28",
    "points": [...],
    "sweeps": [...],
    "bookmarks": {...}
  },
  "summary": {
    "total_points": 150,
    "dimension_range": [2.0, 8.0],
    "peak_findings": {...},
    "critical_crossings": [...]
  }
}
```

## 🎮 Research Workflow Examples

### Discovering Peak Behavior
```python
# 1. Launch research lab
session = enhanced_lab(4.0)

# 2. Explore volume peak region  
results = enhanced_explore(5.26, context="peaks")

# 3. Run detailed sweep around peak
# In lab: sweep 5.2 5.3 100

# 4. Bookmark interesting dimensions
# In lab: bookmark volume_peak_precise

# 5. Export findings
# In lab: export
```

### Mathematical Constant Investigation
```python
# Explore dimensions near π
results = enhanced_explore(3.14159, context="critical")

# Look for golden ratio relationships
results = enhanced_explore(1.618, context="ratios")

# Check e-related dimensions
results = enhanced_explore(2.718, context="derivatives")
```

### Publication Research Pipeline
```bash
# 1. Comprehensive instant analysis
python -m dimensional.cli instant --config publication

# 2. High-resolution sweeps around peaks
python -m dimensional.cli sweep 5.25 5.27 --steps 1000 --export
python -m dimensional.cli sweep 6.33 6.35 --steps 1000 --export
python -m dimensional.cli sweep 7.25 7.27 --steps 1000 --export

# 3. Critical dimension analysis
python -m dimensional.cli explore 5.256431 --context critical --save
python -m dimensional.cli explore 6.335407 --context critical --save
python -m dimensional.cli explore 7.256411 --context critical --save

# 4. Session management and export
python -m dimensional.cli sessions  # Review all data
```

## 🔍 Advanced Features

### Guided Discovery Paths
The system provides intelligent recommendations based on mathematical significance:

- **Peak Proximity**: Automatically detects when exploring near volume, surface, or complexity peaks
- **Critical Constants**: Identifies relationships to π, e, φ, and other mathematical constants  
- **Derivative Analysis**: Finds critical points where rates of change approach zero
- **Pattern Recognition**: Detects interesting ratios and mathematical relationships

### Mathematical Precision
All calculations maintain high precision for research-quality results:

- **Numerical Tolerance**: 1e-12 default precision
- **Error Handling**: Graceful handling of mathematical edge cases
- **Validation**: Input validation prevents invalid mathematical operations
- **Consistency**: All measures use consistent numerical methods

### Integration with Framework
The Enhanced Research CLI seamlessly integrates with the existing dimensional mathematics framework:

- **Backwards Compatibility**: Original `lab()`, `explore()`, `instant()` functions enhanced
- **Fallback Support**: Graceful degradation when enhanced features unavailable
- **Framework Integration**: Access to all dimensional measures (V, S, C) and constants
- **Mathematical Foundation**: Built on proven gamma function and dimensional mathematics

## 💡 Research Tips

1. **Start with `instant()`** to get overview of dimensional landscape
2. **Use `explore()` with context** for focused investigation of specific regions
3. **Run parameter sweeps** around interesting dimensions found during exploration
4. **Bookmark critical dimensions** for easy return in future sessions
5. **Export data regularly** to preserve research progress
6. **Use session persistence** for long-term research projects

## 🎯 Mathematical Discovery Goals

This research system is designed to help discover:

- **The True Shape of One**: Understanding dimensional mathematics relationships
- **Peak Characterization**: Precise location and properties of dimensional maxima
- **Critical Transitions**: Points where mathematical behavior changes
- **Universal Constants**: How π, e, φ appear in dimensional mathematics
- **Pattern Recognition**: Mathematical relationships across dimensional space

The Enhanced Interactive Research CLI transforms dimensional mathematics from computational tool into research discovery platform, providing the rich visualizations, data persistence, and analytical depth needed for serious mathematical research.