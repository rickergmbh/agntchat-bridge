"""Every module in the package must import on the oldest supported Python.

The package is 3.9-clean only because every module carries
`from __future__ import annotations` — without it, a PEP 604 annotation
like `str | None` is evaluated at def time and raises TypeError on import.
That convention held by habit, not by test: two test files had already
drifted off it, and because pytest reported them as *collection errors*
rather than failures, a run that skipped 39 tests still printed a green
"361 passed".

The rest of the suite doesn't close this hole, because it only imports the
modules it happens to exercise. This walks the whole package instead, so a
module nothing else touches still has to load. That is also how
`agentchat.tools.adapters` was found: it imported a `ToolParameter` symbol
deleted from the parent package, and had been unimportable and unreferenced
since the day it landed.

CI runs this on the floor (3.9) and on what the host VMs ship (3.12).
"""

from __future__ import annotations

import importlib
import pkgutil

import pytest

import agentchat

MODULES = sorted(m.name for m in pkgutil.walk_packages(agentchat.__path__, "agentchat."))


def test_package_is_not_empty():
    """Guards the guard: a broken walk would make the test below vacuous."""
    assert len(MODULES) > 20, f"expected the full package, walked only {MODULES}"


@pytest.mark.parametrize("module_name", MODULES)
def test_module_imports(module_name):
    importlib.import_module(module_name)
