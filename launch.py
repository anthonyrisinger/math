#!/usr/bin/env python3
"""
GAMMA EXPLORER - Quick Launch
"""

import subprocess
import sys

modules = {
    "1": ("gamma_lab.py", "🔬 Interactive Gamma Lab - keyboard controls"),
    "2": ("gamma_quick.py", "⚡ Quick tools & one-liners"),
    "3": ("live_gamma.py", "🔥 Live hot-reload editor"),
    "4": ("dim0.py", "🌀 Organic 3D visualization"),
    "5": ("dim2.py", "🎯 Complete framework"),
    "6": ("complexity_peak.py", "📊 V×S complexity peak"),
    "7": ("core_measures.py", "📐 Core geometric measures"),
}

print("\n✨ GAMMA DIMENSIONAL EXPLORER\n")
for k, (f, desc) in modules.items():
    print(f"  [{k}] {desc}")
print("\n  [q] Quit")

choice = input("\n🚀 Launch: ").strip()

if choice in modules:
    file, _ = modules[choice]
    subprocess.run([sys.executable, file])
elif choice != "q":
    print("Invalid choice")
