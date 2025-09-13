import json
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.integration
def test_eval_progress_and_json_no_make(tmp_path: Path):
    # Run eval_agents.py directly to avoid nested make invocation
    cwd = Path(__file__).resolve().parents[1]
    cmd = [
        sys.executable,
        "eval_agents.py",
        "--matches",
        "2",
        "--depth",
        "1",
        "--random-openings",
        "0",
        "--opponent",
        "heuristic",
        "--progress",
    ]
    proc = subprocess.run(
        cmd,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=120,
    )
    out = proc.stdout
    assert proc.returncode == 0, f"eval_agents.py failed:\n{out}"

    # Should show start and progress lines
    assert "[eval] Start:" in out
    assert "[eval] 1/2" in out or "[eval] 2/2" in out

    # Last JSON block should be parseable and contain summary keys
    last_brace = out.rfind("{")
    assert last_brace != -1, f"no JSON found in output:\n{out}"
    summary = json.loads(out[last_brace:])
    for k in ("games", "wins", "losses", "draws", "win_rate"):
        assert k in summary
    assert summary["games"] == 2
