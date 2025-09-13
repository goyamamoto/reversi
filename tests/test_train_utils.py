from train_q import maybe_normalize_features, value_from_features
from eval_params import Params


def test_maybe_normalize_features():
    pos = [4] + [0] * 15
    mob = 20
    stab = 28
    pn, mn, sn = maybe_normalize_features(pos, mob, stab, True, 4.0, 20.0, 28.0)
    assert pn[0] == 1.0
    assert mn == 1.0
    assert sn == 1.0


def test_value_from_features():
    pos = [1] + [0] * 15
    params = Params(pos=[0.0] * 16, mobility_w=2.0, stable_w=3.0)
    params.pos[0] = 5.0
    v = value_from_features(pos, mob=2, stab=3, params=params)
    # 5*1 + 2*2 + 3*3 = 5 + 4 + 9 = 18
    assert v == 18.0
