import json
import os
from pathlib import Path

import pytest

from eval_params import Params
from train_q import _find_latest_weight, _dump_params_json
from eval_agents import load_params_from_candidates, load_params_pool_entries


def test_dump_params_json_contains_meta(tmp_path: Path):
    p = Params.zeros()
    meta = {"note": "test"}
    s = _dump_params_json(p, meta)
    d = json.loads(s)
    assert "__meta__" in d and d["__meta__"]["note"] == "test"
    assert "pos" in d and len(d["pos"]) == 16


def test_find_latest_weight(tmp_path: Path):
    d = tmp_path / "weights"
    d.mkdir()
    f1 = d / "a.json"
    f2 = d / "b.json"
    f1.write_text(json.dumps(Params.zeros().to_dict()))
    f2.write_text(json.dumps(Params.zeros().to_dict()))
    # make f2 newer
    os.utime(f2, None)
    latest = _find_latest_weight(d)
    assert latest == f2


def test_load_params_from_dir_latest(tmp_path: Path):
    d = tmp_path / "weights"
    d.mkdir()
    f1 = d / "w1.json"
    f2 = d / "w2.json"
    p = Params.zeros()
    f1.write_text(json.dumps(p.to_dict()))
    f2.write_text(json.dumps(p.to_dict()))
    # Should pick f2 by mtime
    params = load_params_from_candidates(d)
    assert isinstance(params, Params)


def test_load_params_pool_excludes_given(tmp_path: Path):
    d = tmp_path / "weights"
    d.mkdir()
    f1 = d / "w1.json"
    f2 = d / "w2.json"
    p = Params.zeros()
    f1.write_text(json.dumps(p.to_dict()))
    f2.write_text(json.dumps(p.to_dict()))
    pool = load_params_pool_entries(d, exclude=f2, order="name")
    # Should only return f1
    assert len(pool) == 1 and pool[0][0].name == "w1.json"


def test_load_params_pool_order_newest(tmp_path: Path):
    d = tmp_path / "weights"
    d.mkdir()
    f_old = d / "old.json"
    f_new = d / "new.json"
    p = Params.zeros()
    f_old.write_text(json.dumps(p.to_dict()))
    f_new.write_text(json.dumps(p.to_dict()))
    # Touch new to ensure newer mtime
    import os
    os.utime(f_new, None)
    entries = load_params_pool_entries(d, order="newest")
    assert entries[0][0].name == "new.json"
