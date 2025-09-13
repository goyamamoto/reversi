import pytest

from eval_params import Params
from main import Board, BLACK, WHITE
from main import MinimaxAI


def test_minimax_returns_valid_move_initial():
    b = Board()
    ai = MinimaxAI(depth=1, params=Params.zeros())
    m = ai.choose(b, BLACK)
    assert m is not None
    valids = {(mv.x, mv.y) for mv in b.valid_moves(BLACK)}
    assert (m.x, m.y) in valids


def test_minimax_returns_none_when_no_moves():
    b = Board()
    # fill with WHITE discs so BLACK has no moves
    for y in range(8):
        for x in range(8):
            b.grid[y][x] = WHITE
    ai = MinimaxAI(depth=2, params=Params.zeros())
    assert ai.choose(b, BLACK) is None


@pytest.mark.xfail(reason="Corner preference not guaranteed at shallow depth")
def test_minimax_may_choose_corner_initial():
    b = Board()
    ai = MinimaxAI(depth=1, params=Params.zeros())
    m = ai.choose(b, BLACK)
    assert (m.x, m.y) in {(0, 0), (0, 7), (7, 0), (7, 7)}
