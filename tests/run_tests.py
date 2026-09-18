"""Minimal test runner, so the suite runs with or without pytest.

    python -m tests.run_tests
"""

from __future__ import annotations

import importlib
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    modules = sorted(p.stem for p in Path(__file__).parent.glob("test_*.py"))
    passed, failures = 0, []

    for module_name in modules:
        module = importlib.import_module(f"tests.{module_name}")
        for name in sorted(dir(module)):
            if not name.startswith("test_"):
                continue
            fn = getattr(module, name)
            if not callable(fn):
                continue
            try:
                fn()
            except Exception:
                failures.append((module_name, name, traceback.format_exc()))
                print(f"FAIL  {module_name}.{name}")
            else:
                passed += 1
                print(f"ok    {module_name}.{name}")

    print("-" * 64)
    if failures:
        for module_name, name, tb in failures:
            print(f"\n=== {module_name}.{name} ===\n{tb}")
    print(f"{passed} passed, {len(failures)} failed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
