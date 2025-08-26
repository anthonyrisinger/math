#!/usr/bin/env python3
"""
DIMENSIONAL MATHEMATICS EXPLORER - Quick Launch
"""

import subprocess
import sys

modules = {
    "1": ("dimensional/cli.py", "🎯 Complete CLI interface"),
    "2": (
        "python3 -c 'from dimensional.gamma import demo; demo()'",
        "⚡ Quick gamma demo",
    ),
    "3": (
        "python3 -c 'from dimensional.gamma import lab; lab()'",
        "� Interactive gamma lab",
    ),
    "4": (
        "python3 -c 'from dimensional import instant; instant()'",
        "📊 Instant visualization",
    ),
    "5": ("complexity_peak.py", "📊 V×S complexity peak analysis"),
    "6": ("topo_viz.py", "🌀 3D topological visualization"),
    "7": (
        "python3 -c 'import core; core.print_library_info()'",
        "📐 Core library info",
    ),
}

print("\n✨ DIMENSIONAL MATHEMATICS EXPLORER\n")
for k, (cmd, desc) in modules.items():
    print(f"  [{k}] {desc}")
print("\n  [q] Quit")

choice = input("\n🚀 Launch: ").strip()

if choice in modules:
    cmd, _ = modules[choice]
    if cmd.startswith("python3 -c"):
        subprocess.run(cmd, shell=True)
    else:
        subprocess.run([sys.executable, cmd])
elif choice != "q":
    print("Invalid choice")
