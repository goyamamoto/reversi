from eval_params import evaluate_black_advantage, Params
from main import Board, BLACK, WHITE
from main import MinimaxAI


def flip_colors(board: Board) -> Board:
    nb = board.copy()
    for y in range(8):
        for x in range(8):
            v = nb.grid[y][x]
            if v == BLACK:
                nb.grid[y][x] = WHITE
            elif v == WHITE:
                nb.grid[y][x] = BLACK
    return nb


def test_black_advantage_sign_flips_on_color_swap_stable_only():
    # Construct a position with non-zero edge-stable diff
    b = Board()
    # Clear board
    for y in range(8):
        for x in range(8):
            b.grid[y][x] = 0
    # Top edge: three BLACK from the left corner -> 3 stable for Black
    b.grid[0][0] = BLACK
    b.grid[0][1] = BLACK
    b.grid[0][2] = BLACK
    # Bottom-right corner: one WHITE -> 1 stable for White
    b.grid[7][7] = WHITE

    params = Params(pos=[0.0] * 16, mobility_w=0.0, stable_w=1.0)
    v = evaluate_black_advantage(b, params)
    assert v != 0  # sanity

    bf = flip_colors(b)
    vf = evaluate_black_advantage(bf, params)
    assert vf == -v


def test_minimax_evaluate_matches_black_fixed_sign():
    b = Board()
    # Clear and set a simple asymmetry
    for y in range(8):
        for x in range(8):
            b.grid[y][x] = 0
    b.grid[0][0] = BLACK
    b.grid[7][7] = WHITE

    params = Params(pos=[0.0] * 16, mobility_w=0.0, stable_w=1.0)
    ai = MinimaxAI(depth=1, params=params)
    v_black = ai.evaluate(b, BLACK)
    v_white = ai.evaluate(b, WHITE)
    # evaluate_black_advantage is the black-side value
    vb = evaluate_black_advantage(b, params)
    assert v_black == vb
    assert v_white == -vb

