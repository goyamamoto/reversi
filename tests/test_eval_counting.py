import types

import eval_agents
from eval_agents import evaluate, EvalConfig
from eval_params import Params


def test_evaluate_counts_with_color_swap_monkeypatch():
    # Monkeypatch play_game to return predetermined results:
    # (1,0,0) means Black wins
    seq = [(1, 0, 0), (1, 0, 0)]  # two games, both Black wins
    it = iter(seq)

    def fake_play_game(black_ai, white_ai, random_openings=0, seed=None):
        try:
            return next(it)
        except StopIteration:
            return (0, 0, 1)

    orig = eval_agents.play_game
    eval_agents.play_game = fake_play_game
    try:
        cfg = EvalConfig(matches=2, depth=1, random_openings=0, swap_colors=True)
        # Use heuristic opponent; our params can be None
        res = evaluate(params=None, opponent_kind="heuristic", cfg=cfg, progress=False)
        # First game: our_is_black=True, wins += 1
        # Second game: our_is_black=False, black wins again => our loss
        assert res["games"] == 2
        assert res["wins"] == 1
        assert res["losses"] == 1
        assert res["draws"] == 0
    finally:
        eval_agents.play_game = orig

