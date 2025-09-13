from main import Board, BLACK, WHITE, EMPTY


def test_initial_setup():
    b = Board()
    black, white = b.count()
    assert black == 2
    assert white == 2
    # Check specific starting cells
    assert b.grid[3][3] == WHITE
    assert b.grid[4][4] == WHITE
    assert b.grid[3][4] == BLACK
    assert b.grid[4][3] == BLACK


def test_initial_valid_moves_black():
    b = Board()
    moves = {(m.x, m.y) for m in b.valid_moves(BLACK)}
    expected = {(2, 3), (3, 2), (4, 5), (5, 4)}
    assert moves == expected


def test_apply_move_and_flip():
    b = Board()
    # Black plays at (2,3), should flip (3,3) from WHITE to BLACK
    move = next(m for m in b.valid_moves(BLACK) if (m.x, m.y) == (2, 3))
    b.apply_move(move, BLACK)
    assert b.grid[3][2] == BLACK
    assert b.grid[3][3] == BLACK


def test_copy_independent():
    b1 = Board()
    b2 = b1.copy()
    assert id(b1.grid) != id(b2.grid)
    b1.grid[0][0] = BLACK
    assert b2.grid[0][0] == EMPTY


def test_full_board():
    b = Board()
    # Fill the board with BLACK
    for y in range(8):
        for x in range(8):
            b.grid[y][x] = BLACK
    assert b.full()
    assert b.valid_moves(BLACK) == []
    assert b.valid_moves(WHITE) == []


def test_pass_scenario():
    # Construct a position where WHITE has a legal move but BLACK has none
    b = Board()
    # Set all to WHITE first
    for y in range(8):
        for x in range(8):
            b.grid[y][x] = WHITE
    # Make (2,0) empty and (1,0) BLACK, keep (3,0) WHITE to enable WHITE move at (2,0)
    b.grid[0][2] = EMPTY
    b.grid[0][1] = BLACK
    b.grid[0][3] = WHITE

    black_moves = b.valid_moves(BLACK)
    white_moves = b.valid_moves(WHITE)
    assert len(black_moves) == 0, "BLACK should have no legal moves (pass)"
    assert len(white_moves) >= 1, "WHITE should have at least one legal move"
