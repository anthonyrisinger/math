#!/usr/bin/env python3
"""
Setup script for Dimensional Mathematics Project
================================================

Install with:
    pip install -e .                # Development install
    pip install .                   # Regular install
    pip install -e .[dev]           # With development dependencies
    pip install -e .[docs]          # With documentation dependencies
    pip install -e .[all]           # Everything
"""

import os

from setuptools import find_packages, setup


# Read requirements
def read_requirements(filename):
    """Read requirements from file."""
    path = os.path.join(os.path.dirname(__file__), filename)
    if os.path.exists(path):
        with open(path) as f:
            return [
                line.strip() for line in f if line.strip() and not line.startswith("#")
            ]
    return []


# Read long description
def read_long_description():
    """Read long description from README."""
    readme_path = os.path.join(os.path.dirname(__file__), "README.md")
    if os.path.exists(readme_path):
        with open(readme_path, encoding="utf-8") as f:
            return f.read()
    return "Dimensional Mathematics - unified mathematical modeling framework"


# Core requirements
CORE_REQUIREMENTS = [
    "numpy>=1.21.0",
    "scipy>=1.7.0",
    "matplotlib>=3.5.0",
    "click>=8.0.0",
    "pydantic>=2.0.0",
    "rich>=13.0.0",
    "typer>=0.9.0",
    "plotly>=5.0.0",
    "bokeh>=3.0.0",
]  # Development requirements
DEV_REQUIREMENTS = [
    "pytest>=6.0.0",
    "pytest-cov>=3.0.0",
    "jupyter>=1.0.0",
    "ipywidgets>=7.6.0",
]

# Documentation requirements
DOC_REQUIREMENTS = [
    "sphinx>=4.0.0",
    "sphinx-rtd-theme>=1.0.0",
    "myst-parser>=0.17.0",
]

# Advanced/optional requirements
ADVANCED_REQUIREMENTS = [
    "sympy>=1.9.0",
    "plotly>=5.0.0",
    "panel>=0.13.0",
    "param>=1.12.0",
]

setup(
    # Package metadata
    name="dimensional-mathematics",
    version="1.0.0",
    author="Dimensional Mathematics Project",
    author_email="contact@dimensional-math.org",
    description="Unified mathematical modeling framework for dimensional emergence theory",
    long_description=read_long_description(),
    long_description_content_type="text/markdown",
    url="https://github.com/dimensional-mathematics/framework",
    # Package structure
    packages=find_packages(),
    include_package_data=True,
    # Requirements
    install_requires=CORE_REQUIREMENTS,
    extras_require={
        "dev": DEV_REQUIREMENTS,
        "docs": DOC_REQUIREMENTS,
        "advanced": ADVANCED_REQUIREMENTS,
        "all": DEV_REQUIREMENTS + DOC_REQUIREMENTS + ADVANCED_REQUIREMENTS,
    },
    # Python version
    python_requires=">=3.8",
    # Entry points
    entry_points={
        "console_scripts": [
            "dimensional=dimensional.cli:main",
            "dim=dimensional.cli:main",
        ],
    },
    # Classification
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Education",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Visualization",
    ],
    # Keywords
    keywords=[
        "mathematics",
        "gamma-function",
        "dimensional-analysis",
        "phase-dynamics",
        "topology",
        "visualization",
        "interactive",
    ],
    # Project URLs
    project_urls={
        "Documentation": "https://dimensional-mathematics.readthedocs.io/",
        "Source": "https://github.com/dimensional-mathematics/framework",
        "Tracker": "https://github.com/dimensional-mathematics/framework/issues",
    },
    # Package data
    package_data={
        "dimensional": ["*.md", "*.txt", "*.yml"],
    },
    # Zip safety
    zip_safe=False,
)
