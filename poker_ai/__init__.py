from __future__ import annotations

import logging

from rich.logging import RichHandler

FORMAT = "%(message)s"
logging.basicConfig(
    format=FORMAT, datefmt="[%X] ", handlers=[RichHandler()], level=logging.INFO,
)

def __getattr__(name):
    """Lazy subpackage imports — only load when accessed."""
    import importlib
    _subpackages = {"ai", "cli", "clustering", "games", "poker", "terminal", "utils"}
    if name in _subpackages:
        return importlib.import_module(f".{name}", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__version__ = "1.0.0rc3"
