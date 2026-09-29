import importlib

import sdgf

SUBPACKAGES = [
    "spec",
    "tasktypes",
    "models",
    "coverage",
    "generate",
    "tools",
    "governance",
    "validate",
    "judge",
    "evaluation",
    "store",
    "hitl",
]


def test_package_imports():
    assert sdgf.__version__


def test_subpackages_import():
    for name in SUBPACKAGES:
        importlib.import_module(f"sdgf.{name}")
    importlib.import_module("sdgf.pipeline")
    importlib.import_module("sdgf.cli")
