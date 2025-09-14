from main import Board, BLACK, WHITE
from eval_params import Params, evaluate_full_black_advantage
from train_q import (
    minimax_value_black_adv,
    choose_minimax_move,
    epsilon_greedy_action,
    _leaf_value_black_adv,
)


def make_full(color: int) -> Board:
    b = Board()
    for y in range(8):
        for x in range(8):
            b.grid[y][x] = color
    return b


def test_minimax_terminal_full_black_equals_full_value():
    params = Params.zeros()
    params.stable_w = 1.0
    b = make_full(BLACK)
    v = minimax_value_black_adv(
        board=b,
        params=params,
        depth=3,
        to_move=BLACK,
        normalize=True,
        pos_scale=4.0,
        mob_scale=20.0,
        stab_scale=28.0,
    )
    assert v == evaluate_full_black_advantage(params)


def test_leaf_value_white_win_negative_full():
    params = Params.zeros()
    params.stable_w = 1.0
    b = make_full(WHITE)
    v = _leaf_value_black_adv(b, params, True, 4.0, 20.0, 28.0)
    assert v == -evaluate_full_black_advantage(params)


def test_choose_minimax_move_returns_legal_initial():
    params = Params.zeros()
    b = Board()
    m_b = choose_minimax_move(b, BLACK, params, depth=1, normalize=True, pos_scale=4.0, mob_scale=20.0, stab_scale=28.0)
    assert m_b is not None
    valids_b = {(mv.x, mv.y) for mv in b.valid_moves(BLACK)}
    assert (m_b.x, m_b.y) in valids_b

    m_w = choose_minimax_move(b, WHITE, params, depth=1, normalize=True, pos_scale=4.0, mob_scale=20.0, stab_scale=28.0)
    assert m_w is not None
    valids_w = {(mv.x, mv.y) for mv in b.valid_moves(WHITE)}
    assert (m_w.x, m_w.y) in valids_w


def test_epsilon_greedy_action_random_move_initial():
    params = Params.zeros()
    b = Board()
    m = epsilon_greedy_action(
        board=b,
        player=BLACK,
        params=params,
        epsilon=1.0,  # always random
        normalize_features_flag=True,
        pos_scale=4.0,
        mob_scale=20.0,
        stab_scale=28.0,
    )
    assert m is not None
    valids = {(mv.x, mv.y) for mv in b.valid_moves(BLACK)}
    assert (m.x, m.y) in valids

