from __future__ import annotations

from importlib import import_module
from pathlib import Path

# Allow pysrat.data to behave as a package that can load submodules
__path__ = [str(Path(__file__).with_name("data"))]

NHPPData = import_module("pysrat.data.nhpp").NHPPData

__all__ = ["NHPPData"]
