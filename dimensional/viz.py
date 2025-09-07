"""
Simplified visualization module - works with basic dependencies.
"""

import numpy as np

from .core import c, gamma, r, s, v


def explore(d: float = 4.0, **kwargs):
    """
    Explore dimensional measures at dimension d.

    Returns dictionary with all computed values.
    """
    results = {
        "dimension": d,
        "volume": v(d),
        "surface": s(d),
        "complexity": c(d),
        "ratio": r(d),
        "gamma": gamma(d) if d > 0 else None,
    }

    # Pretty print results
    print(f"\n{'='*40}")
    print(f"  Dimension d = {d}")
    print(f"{'='*40}")
    print(f"  Volume:     {results['volume']:.6f}")
    print(f"  Surface:    {results['surface']:.6f}")
    print(f"  Complexity: {results['complexity']:.6f}")
    print(f"  Ratio:      {results['ratio']:.6f}")
    if results['gamma']:
        print(f"  Gamma:      {results['gamma']:.6f}")
    print(f"{'='*40}\n")

    return results


def instant(d_range=None):
    """
    Show all measures across dimension range.
    """
    if d_range is None:
        d_range = np.linspace(0.1, 20, 10)
    else:
        d_range = np.asarray(d_range)

    print(f"\n{'='*60}")
    print(f"{'Dimension':>10} {'Volume':>12} {'Surface':>12} {'Complexity':>12}")
    print(f"{'='*60}")

    for d in d_range:
        vol = v(d)
        surf = s(d)
        comp = c(d)
        print(f"{d:10.2f} {vol:12.4e} {surf:12.4e} {comp:12.4e}")

    print(f"{'='*60}\n")

    return {
        "dimensions": d_range.tolist(),
        "volume": v(d_range).tolist(),
        "surface": s(d_range).tolist(),
        "complexity": c(d_range).tolist()
    }


def lab(d=4.0):
    """
    Simple laboratory interface.
    """
    print("\n🔬 Dimensional Mathematics Laboratory")
    print("="*40)

    result = explore(d, show_plot=False)

    # Find peaks
    peak_info = peaks()
    print("\n📊 Key Dimensions (Peaks):")
    for measure, (dim, val) in peak_info.items():
        print(f"  {measure}: d={dim:.3f} (value={val:.4e})")

    return result


def peaks():
    """
    Find peaks in dimensional measures.
    """
    from scipy.optimize import minimize_scalar

    # Find volume peak
    vol_result = minimize_scalar(lambda d: -v(d), bounds=(1, 10), method='bounded')
    vol_peak = (vol_result.x, -vol_result.fun)

    # Find surface peak
    surf_result = minimize_scalar(lambda d: -s(d), bounds=(1, 12), method='bounded')
    surf_peak = (surf_result.x, -surf_result.fun)

    # Find complexity peak
    comp_result = minimize_scalar(lambda d: -c(d), bounds=(1, 15), method='bounded')
    comp_peak = (comp_result.x, -comp_result.fun)

    return {
        "volume": vol_peak,
        "surface": surf_peak,
        "complexity": comp_peak
    }


def demo():
    """
    Run a demonstration of the package capabilities.
    """
    print("\n🎯 Dimensional Mathematics Demo")
    print("="*40)

    # Show interesting dimensions
    interesting = [2, 3, 4, 5.257, 7.257]

    for d in interesting:
        print(f"\nDimension {d}:")
        explore(d, show_plot=False)

    print("\n💡 Key Features:")
    print("  • explore(d) - Explore any dimension")
    print("  • instant()  - See patterns across dimensions")
    print("  • lab()      - Interactive laboratory")
    print("  • peaks()    - Find optimal dimensions")

    return True
