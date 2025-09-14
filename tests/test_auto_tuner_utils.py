from pathlib import Path
import json
import os

from auto_tuner import list_final_versions, previous_final_version, previous_final_versions
from eval_params import Params


def write_weight(path: Path):
    d = Params.zeros().to_dict()
    d["__meta__"] = {"note": "test"}
    path.write_text(json.dumps(d))


def test_previous_final_version_picks_second_newest(tmp_path: Path):
    d = tmp_path / "weights"
    d.mkdir()
    f1 = d / "weights-20250101-000000-1000ep.json"
    f2 = d / "weights-20250102-000000-1000ep.json"
    f3 = d / "weights-20250103-000000-1000ep.json"
    for f in (f1, f2, f3):
        write_weight(f)
    # Make mtimes increasing
    os.utime(f1, (f1.stat().st_atime, f1.stat().st_mtime - 200))
    os.utime(f2, (f2.stat().st_atime, f2.stat().st_mtime - 100))
    # list_final_versions returns newest first
    versions = list_final_versions(d)
    assert versions[0].name == f3.name
    assert versions[1].name == f2.name
    assert versions[2].name == f1.name
    prev = previous_final_version(d)
    assert prev is not None and prev.name == f2.name
    prevs = previous_final_versions(d, k=2)
    assert [p.name for p in prevs] == [f2.name, f1.name]
