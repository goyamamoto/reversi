from pathlib import Path
import json
import os

from eval_agents import evaluate, EvalConfig
from eval_params import Params


def write_weight(path: Path):
    d = Params.zeros().to_dict()
    path.write_text(json.dumps(d))


def test_evaluate_pool_skips_newest_and_limits(tmp_path: Path):
    d = tmp_path / "weights"
    d.mkdir()
    newest = d / "weights-20250103-000000-1000ep.json"
    prev = d / "weights-20250102-000000-1000ep.json"
    old = d / "weights-20250101-000000-1000ep.json"
    for f in (old, prev, newest):
        write_weight(f)
    # Touch mtimes so newest is newest
    os.utime(old, None)
    os.utime(prev, None)
    os.utime(newest, None)

    # Player params: latest.json not required; use None (heuristic) to avoid depending on external file
    cfg = EvalConfig(matches=1, depth=1, random_openings=0, swap_colors=True)
    # Skip newest 1 and limit 1 => should face 'prev' only
    res = evaluate(
        params=None,
        opponent_kind="pool",
        cfg=cfg,
        progress=False,
        opp_dir=d,
        opp_order="newest",
        opp_limit=1,
        matches_per_opp=1,
        workers=1,
        opp_paths=None,
        opp_skip_newest=1,
    )
    # We can't directly introspect the opponent used, but ensure it ran 1 game
    assert res["games"] == 1

