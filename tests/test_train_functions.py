from main import Board, BLACK, WHITE
from eval_params import Params, evaluate_full_black_advantage
from train_q import (
    _feature_vector_black,
    features_state_diff_black,
    maybe_normalize_features,
    value_from_features,
    _leaf_value_black_adv,
)


def test_feature_vector_black_sign_flip():
    b = Board()
    # clear board
    for y in range(8):
        for x in range(8):
            b.grid[y][x] = 0
    b.grid[0][0] = BLACK

    vec_black = _feature_vector_black(b, BLACK, True, 4.0, 20.0, 28.0)
    vec_white = _feature_vector_black(b, WHITE, True, 4.0, 20.0, 28.0)
    assert len(vec_black) == 18
    assert len(vec_white) == 18
    # white turn vector should be exact negative of black
    for vb, vw in zip(vec_black, vec_white):
        assert vw == -vb


def test_value_from_features_linear_combination():
    params = Params(pos=[0.0] * 16, mobility_w=2.0, stable_w=3.0)
    # pos-weight[0]=5 contributes 5, mobility=2 contributes 4, stable=3 contributes 9 => total 18
    params.pos[0] = 5.0
    pos = [0.0] * 16
    pos[0] = 1.0
    v = value_from_features(pos, 2.0, 3.0, params)
    assert v == 18.0


def test_maybe_normalize_features_scales():
    pos = [4.0] + [0.0] * 15
    mob = 20.0
    stab = 28.0
    pn, mn, sn = maybe_normalize_features(pos, mob, stab, True, 4.0, 20.0, 28.0)
    assert pn[0] == 1.0 and all(x == 0.0 for x in pn[1:])
    assert mn == 1.0
    assert sn == 1.0


def test_leaf_value_black_adv_terminal_matches_full_black():
    # Full black board => leaf value equals full-black advantage value
    b = Board()
    for y in range(8):
        for x in range(8):
            b.grid[y][x] = BLACK
    params = Params.zeros()
    # give stable a weight so it's nonzero
    params.stable_w = 1.0
    v_leaf = _leaf_value_black_adv(b, params, True, 4.0, 20.0, 28.0)
    v_full = evaluate_full_black_advantage(params)
    assert v_leaf == v_full
