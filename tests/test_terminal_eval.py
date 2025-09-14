from eval_params import evaluate_black_advantage, evaluate_full_black_advantage, Params
from main import Board, BLACK, WHITE


def make_full(color: int) -> Board:
    b = Board()
    for y in range(8):
        for x in range(8):
            b.grid[y][x] = color
    return b


def test_terminal_full_black_equals_full_black_eval():
    params = Params(pos=[0.0] * 16, mobility_w=0.0, stable_w=1.0)
    b = make_full(BLACK)
    v = evaluate_black_advantage(b, params)
    fb = evaluate_full_black_advantage(params)
    assert v == fb


def test_terminal_full_white_equals_negative_full_black_eval():
    params = Params(pos=[0.0] * 16, mobility_w=0.0, stable_w=1.0)
    b = make_full(WHITE)
    v = evaluate_black_advantage(b, params)
    fb = evaluate_full_black_advantage(params)
    assert v == -fb


def test_terminal_draw_returns_zero():
    params = Params(pos=[0.0] * 16, mobility_w=0.0, stable_w=1.0)
    b = Board()
    # Fill half black, half white (draw) – no legal moves because full
    toggle = BLACK
    for y in range(8):
        for x in range(8):
            b.grid[y][x] = toggle
            toggle = WHITE if toggle == BLACK else BLACK
    v = evaluate_black_advantage(b, params)
    assert v == 0.0

