#!/usr/bin/env python3
"""Integration tests to prevent architectural regressions."""

import ast
import importlib
import sys
from pathlib import Path
from typing import Dict, List, Set

import pytest


class TestArchitecturalIntegrity:
    """Tests that catch real architectural problems."""

    def test_no_circular_imports(self):
        """Ensure no circular import dependencies exist."""
        # This test actually caught the gamma.py ↔ research_cli.py cycle
        root = Path("dimensional")
        imports = self._build_import_graph(root)
        cycles = self._find_cycles(imports)

        assert len(cycles) == 0, f"Circular imports detected: {cycles}"

    def test_layer_hierarchy_enforced(self):
        """Ensure modules only import from lower layers."""
        # Define our actual layer hierarchy
        layers = {
            0: ["logger", "validation", "mathematics.constants", "mathematics.functions"],
            1: ["measures", "morphic"],
            2: ["gamma", "phase"],
            3: ["research_cli", "performance"],
            4: ["cli", "interface"]
        }

        # Build reverse mapping
        module_to_layer = {}
        for layer, modules in layers.items():
            for module in modules:
                module_to_layer[f"dimensional.{module}"] = layer

        # Check each module only imports from same or lower layers
        violations = []
        root = Path("dimensional")
        imports = self._build_import_graph(root)

        for module, deps in imports.items():
            if module in module_to_layer:
                module_layer = module_to_layer[module]
                for dep in deps:
                    if dep in module_to_layer:
                        dep_layer = module_to_layer[dep]
                        if dep_layer > module_layer:
                            violations.append(
                                f"{module} (layer {module_layer}) imports "
                                f"{dep} (layer {dep_layer})"
                            )

        assert len(violations) == 0, f"Layer violations: {violations}"

    def test_all_modules_importable(self):
        """Ensure all modules can be imported without errors."""
        # This catches import-time errors including some circular dependencies
        modules_to_test = [
            "dimensional.gamma",
            "dimensional.measures",
            "dimensional.morphic",
            "dimensional.performance",
            "dimensional.research_cli",
            "dimensional.phase",
            "dimensional.interface",
        ]

        failed = []
        for module in modules_to_test:
            try:
                # Clear from sys.modules to force fresh import
                if module in sys.modules:
                    del sys.modules[module]
                importlib.import_module(module)
            except Exception as e:
                failed.append(f"{module}: {e}")

        assert len(failed) == 0, f"Import failures: {failed}"

    def test_no_cross_test_imports(self):
        """Ensure test files don't import from each other."""
        # Tests should be independent
        test_dir = Path("tests")
        test_imports = {}

        for py_file in test_dir.rglob("test_*.py"):
            imports = self._extract_imports(py_file)
            test_imports[str(py_file)] = imports

        violations = []
        for test_file, imports in test_imports.items():
            for imp in imports:
                if "test_" in imp and imp != test_file:
                    violations.append(f"{test_file} imports {imp}")

        assert len(violations) == 0, f"Test cross-imports: {violations}"

    def _build_import_graph(self, root: Path) -> Dict[str, Set[str]]:
        """Build import dependency graph."""
        graph = {}

        for py_file in root.rglob("*.py"):
            if "__pycache__" in str(py_file):
                continue
            if "test_" in py_file.name:
                continue

            module = str(py_file.relative_to(root.parent).with_suffix('')).replace('/', '.')
            imports = self._extract_imports(py_file)

            # Filter to internal imports only
            internal = set()
            for imp in imports:
                if imp.startswith("dimensional"):
                    internal.add(imp)

            graph[module] = internal

        return graph

    def _extract_imports(self, filepath: Path) -> Set[str]:
        """Extract all imports from a Python file."""
        imports = set()

        try:
            with open(filepath) as f:
                tree = ast.parse(f.read())

            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for name in node.names:
                        imports.add(name.name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imports.add(node.module)
        except:
            pass

        return imports

    def _find_cycles(self, graph: Dict[str, Set[str]]) -> List[List[str]]:
        """Find all cycles in import graph using DFS."""
        cycles = []
        visited = set()
        rec_stack = []

        def dfs(node):
            if node in rec_stack:
                # Found cycle
                cycle_start = rec_stack.index(node)
                cycle = rec_stack[cycle_start:] + [node]
                cycles.append(cycle)
                return

            if node in visited:
                return

            visited.add(node)
            rec_stack.append(node)

            for neighbor in graph.get(node, []):
                dfs(neighbor)

            rec_stack.pop()

        for node in graph:
            if node not in visited:
                dfs(node)

        # Deduplicate cycles
        unique = []
        seen = set()
        for cycle in cycles:
            if len(cycle) > 1:
                # Normalize cycle
                min_idx = cycle.index(min(cycle))
                normalized = tuple(cycle[min_idx:] + cycle[:min_idx])
                if normalized not in seen:
                    seen.add(normalized)
                    unique.append(list(normalized))

        return unique


class TestModuleLoadOrder:
    """Test that modules can load in any order without circular import issues."""

    def test_reverse_import_order(self):
        """Import modules in reverse dependency order."""
        # If there are hidden circular dependencies, this will expose them
        import_order = [
            "dimensional.cli",
            "dimensional.research_cli",
            "dimensional.gamma",
            "dimensional.measures",
            "dimensional.mathematics.functions",
        ]

        for module in import_order:
            if module in sys.modules:
                del sys.modules[module]

        # Now import in reverse order - should still work
        for module in import_order:
            try:
                importlib.import_module(module)
            except ImportError as e:
                pytest.fail(f"Circular dependency exposed: {module} - {e}")

    def test_parallel_import_safety(self):
        """Ensure modules are import-safe for parallel execution."""
        # This would catch modules with import-time side effects
        from concurrent.futures import ThreadPoolExecutor

        modules = [
            "dimensional.measures",
            "dimensional.gamma",
            "dimensional.morphic",
            "dimensional.phase",
        ]

        def import_module(name):
            if name in sys.modules:
                del sys.modules[name]
            return importlib.import_module(name)

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(import_module, m) for m in modules]
            results = [f.result() for f in futures]

        assert len(results) == len(modules), "Parallel import failed"
