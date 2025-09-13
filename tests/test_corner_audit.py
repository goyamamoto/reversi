from eval_agents import corner_audit, EvalConfig
from eval_params import Params


def test_corner_audit_runs_with_defaults():
    # no params (heuristic) vs random, 1 game, depth 1, no random openings
    cfg = EvalConfig(matches=1, depth=1, random_openings=0, swap_colors=True)
    res = corner_audit(params=None, opponent_kind="random", cfg=cfg, progress=False)
    assert set(res.keys()) == {"opportunities", "taken", "rate", "games"}
    assert res["games"] == 1

