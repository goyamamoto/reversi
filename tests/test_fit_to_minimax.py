import types

import pytest

import train_q
from eval_params import Params


@pytest.fixture
def patch_minimax_and_feature(monkeypatch):
    # Save originals
    orig_minimax = train_q.minimax_value_black_adv
    orig_feat = train_q._feature_vector_black

    def fake_minimax(board, params, depth, to_move, normalize, pos_scale, mob_scale, stab_scale, workers=1):
        # Constant teacher value in black-advantage scalar
        return 3.0

    def fake_feature(board, to_move, normalize, pos_scale, mob_scale, stab_scale):
        # 18-dim: unit in first component, signed by turn (BLACK=+1, WHITE=-1)
        sign = 1.0 if to_move == train_q.BLACK else -1.0
        vec = [0.0] * 18
        vec[0] = sign * 1.0
        return vec

    monkeypatch.setattr(train_q, "minimax_value_black_adv", fake_minimax)
    monkeypatch.setattr(train_q, "_feature_vector_black", fake_feature)

    yield

    # restore
    monkeypatch.setattr(train_q, "minimax_value_black_adv", orig_minimax)
    monkeypatch.setattr(train_q, "_feature_vector_black", orig_feat)


def test_fit_to_minimax_solves_linear_weights(patch_minimax_and_feature):
    params0 = Params.zeros()
    out = train_q.fit_to_minimax(
        episodes=5,
        params=params0,
        depth=2,
        normalize=True,
        pos_scale=4.0,
        mob_scale=20.0,
        stab_scale=28.0,
        random_openings=0,
        l2=1e-6,
        ridge_to_current=False,
        workers=1,
        max_samples=50,
        play_policy="random",
        sample_epsilon=1.0,
    )
    # Expect the first positional weight to match teacher 3.0, others near 0
    assert pytest.approx(out.pos[0], rel=1e-6, abs=1e-6) == 3.0
    for i in range(1, 16):
        assert abs(out.pos[i]) < 1e-6
    assert abs(out.mobility_w) < 1e-6
    assert abs(out.stable_w) < 1e-6


def test_fit_to_minimax_ridge_to_current_bias(patch_minimax_and_feature):
    # Start from weight 5.0; with large l2 ridge-to-current, solution should be closer to 5 than to 3
    params0 = Params(pos=[5.0] + [0.0] * 15, mobility_w=0.0, stable_w=0.0)
    out = train_q.fit_to_minimax(
        episodes=3,
        params=params0,
        depth=2,
        normalize=True,
        pos_scale=4.0,
        mob_scale=20.0,
        stab_scale=28.0,
        random_openings=0,
        l2=1e2,
        ridge_to_current=True,
        workers=1,
        max_samples=30,
        play_policy="random",
        sample_epsilon=1.0,
    )
    # out.pos[0] should be between 3 and 5, and closer to 5 due to ridge to current
    assert 3.0 <= out.pos[0] <= 5.0
    assert abs(out.pos[0] - 5.0) < abs(out.pos[0] - 3.0)

