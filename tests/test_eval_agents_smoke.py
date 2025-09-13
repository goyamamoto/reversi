import json

from eval_agents import evaluate, EvalConfig, RandomAI, play_game
from eval_params import Params
from main import MinimaxAI


def test_evaluate_random_opponent_counts_sum():
    cfg = EvalConfig(matches=6, depth=1, random_openings=0, swap_colors=True, seed=123)
    res = evaluate(params=None, opponent_kind="random", cfg=cfg)
    assert res["games"] == 6
    assert res["wins"] + res["losses"] + res["draws"] == 6


def test_play_game_result_tuple():
    w, l, d = play_game(RandomAI(), RandomAI(), random_openings=0, seed=42)
    assert (w, l, d) in {(1, 0, 0), (0, 1, 0), (0, 0, 1)}

