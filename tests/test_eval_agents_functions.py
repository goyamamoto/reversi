from pathlib import Path
import json
import os

from eval_agents import load_params_pool_entries, load_params_from_paths
from eval_params import Params


def write_weight(path: Path):
    d = Params.zeros().to_dict()
    path.write_text(json.dumps(d))


def test_load_params_pool_entries_order_newest(tmp_path: Path):
    d = tmp_path / "weights"
    d.mkdir()
    old = d / "w1.json"
    mid = d / "w2.json"
    new = d / "w3.json"
    for f in (old, mid, new):
        write_weight(f)
        os.utime(f, None)
    entries = load_params_pool_entries(d, order="newest")
    names = [p.name for p, _ in entries]
    assert names[0] == "w3.json"


def test_load_params_from_paths_preserves_length(tmp_path: Path):
    a = tmp_path / "a.json"
    b = tmp_path / "b.json"
    write_weight(a)
    write_weight(b)
    entries = load_params_from_paths([a, b])
    assert len(entries) == 2

