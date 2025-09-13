import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.integration
def test_auto_tune_progress_smoke(tmp_path: Path):
    # Keep it tiny to run fast
    cmd = [
        sys.executable,
        "auto_tuner.py",
        "--cycles",
        "0",  # baseline only to avoid lengthy training
        "--episodes",
        "1",
        "--matches",
        "1",
        "--random-openings-eval",
        "0",
        "--normalize-features",
        "--progress",
    ]
    env = os.environ.copy()
    # Unbuffered for immediate output
    env["PYTHONUNBUFFERED"] = "1"
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=120)
    out = proc.stdout
    assert proc.returncode == 0, f"auto_tuner.py failed:\n{out}"
    # Expect baseline and evaluation progress lines
    assert "[auto-tune] Start:" in out
    assert ("[auto-tune] Baseline evaluating:" in out) or ("[eval]" in out)
