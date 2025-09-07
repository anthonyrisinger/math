#!/usr/bin/env python3
"""
Command-line interface for dimensional mathematics.
"""

import sys

from . import explore, instant, lab, peaks, s, v


def main():
    """Main CLI entry point."""
    if len(sys.argv) < 2:
        print_help()
        return

    command = sys.argv[1].lower()

    if command in ['help', '-h', '--help']:
        print_help()
    elif command == 'explore':
        d = float(sys.argv[2]) if len(sys.argv) > 2 else 4.0
        explore(d)
    elif command == 'instant':
        instant()
    elif command == 'lab':
        lab()
    elif command == 'peaks':
        peak_results = peaks()
        print("\n📊 Dimensional Peaks:")
        for measure, (dim, val) in peak_results.items():
            print(f"  {measure}: d={dim:.3f} (value={val:.4e})")
    elif command == 'v':
        d = float(sys.argv[2]) if len(sys.argv) > 2 else 4.0
        print(f"V({d}) = {v(d):.6f}")
    elif command == 's':
        d = float(sys.argv[2]) if len(sys.argv) > 2 else 4.0
        print(f"S({d}) = {s(d):.6f}")
    elif command == 'demo':
        from .viz import demo
        demo()
    else:
        print(f"Unknown command: {command}")
        print_help()

def print_help():
    """Print help message."""
    print("""
🔮 Dimensional Mathematics Explorer

Usage: python -m dimensional <command> [options]

Commands:
  explore [d]  - Explore dimension d (default: 4)
  instant      - Visualize measures across dimensions
  lab          - Interactive laboratory
  peaks        - Find peak dimensions
  demo         - Run demonstration
  v [d]        - Calculate volume at dimension d
  s [d]        - Calculate surface area at dimension d
  help         - Show this help

Examples:
  python -m dimensional explore 5
  python -m dimensional peaks
  python -m dimensional demo
""")

if __name__ == "__main__":
    main()
