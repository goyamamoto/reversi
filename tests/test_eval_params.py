from eval_params import sym_index, compute_pos_features, count_edge_stable, evaluate_with_params, Params
from main import Board, BLACK, WHITE, EMPTY


def test_symmetry_indices():
    # corners map to same index
    idxs = {
        sym_index(0, 0),
        sym_index(7, 0),
        sym_index(0, 7),
        sym_index(7, 7),
    }
    assert len(idxs) == 1


def test_compute_pos_features_signs():
    b = Board()
    # clear board
    for y in range(8):
        for x in range(8):
            b.grid[y][x] = EMPTY
    b.grid[0][0] = BLACK
    b.grid[7][7] = WHITE
    feats_black = compute_pos_features(b.grid, BLACK)
    feats_white = compute_pos_features(b.grid, WHITE)
    # (0,0) and (7,7) both map to same quadrant index 0
    assert feats_black[0] == 0  # +1 (black) and -1 (white) cancel
    assert feats_white[0] == 0


def test_count_edge_stable():
    b = Board()
    # make top edge black, rest empty
    for y in range(8):
        for x in range(8):
            b.grid[y][x] = EMPTY
    for x in range(8):
        b.grid[0][x] = BLACK
    assert count_edge_stable(b.grid, BLACK) == 8
    assert count_edge_stable(b.grid, WHITE) == 0


def test_evaluate_with_params_linear():
    b = Board()
    # clear board
    for y in range(8):
        for x in range(8):
            b.grid[y][x] = EMPTY
    b.grid[0][0] = BLACK
    params = Params(pos=[0.0] * 16, mobility_w=0.0, stable_w=0.0)
    params.pos[0] = 1.0  # weight only quadrant index 0
    v_black = evaluate_with_params(b, BLACK, params)
    v_white = evaluate_with_params(b, WHITE, params)
    assert v_black == 1.0
    assert v_white == -1.0
