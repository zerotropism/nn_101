"""Smoke test: the script runs end to end without raising.

main.py executes everything at module level (no __main__ guard), so it is run
as a subprocess rather than imported. Restructuring it into importable
functions belongs to the ml-foundations refactor (step 3.7).
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_script_runs() -> None:
    result = subprocess.run(
        [sys.executable, str(ROOT / "main.py")],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert "neuron forward pass" in result.stdout
