"""Test basic environment dependencies and package imports."""
import importlib
import pytest


def test_core_imports():
    """Verify standard scientific stack is importable."""
    packages = ["numpy", "scipy", "pandas", "yaml", "matplotlib", "pytest"]
    for pkg in packages:
        mod = importlib.import_module(pkg)
        assert mod is not None, f"Failed to import {pkg}"


def test_package_import():
    """Verify edge_switching package is importable."""
    import edge_switching
    assert hasattr(edge_switching, "__version__")
    assert edge_switching.__version__ == "0.1.0"


def test_plot_style_readable():
    """Verify matplotlib plot style exists and is readable."""
    from pathlib import Path
    style_path = Path(__file__).resolve().parent.parent / "configs" / "plot_style.mplstyle"
    assert style_path.exists(), f"Plot style file not found at {style_path}"
