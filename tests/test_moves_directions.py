from main import Board, BLACK, WHITE, EMPTY


def test_horizontal_multi_flip():
    b = Board()
    # clear
    for y in range(8):
        for x in range(8):
            b.grid[y][x] = EMPTY
    # Row 3: B W W W W W W .  -> Black can play at (7,3) flipping 6
    b.grid[3][0] = BLACK
    for x in range(1, 7):
        b.grid[3][x] = WHITE
    # (7,3) is empty
    mv = next(m for m in b.valid_moves(BLACK) if (m.x, m.y) == (7, 3))
    assert len(mv.flips) == 6
    assert set(mv.flips) == {(x, 3) for x in range(1, 7)}


def test_diagonal_multi_flip():
    b = Board()
    # clear
    for y in range(8):
        for x in range(8):
            b.grid[y][x] = EMPTY
    # Diagonal: (0,0)=B, (1,1..6)=W, (7,7)=empty -> move at (7,7)
    b.grid[0][0] = BLACK
    for i in range(1, 7):
        b.grid[i][i] = WHITE
    mv = next(m for m in b.valid_moves(BLACK) if (m.x, m.y) == (7, 7))
    assert len(mv.flips) == 6
    assert set(mv.flips) == {(i, i) for i in range(1, 7)}


def test_invalid_without_bracket():
    b = Board()
    # clear
    for y in range(8):
        for x in range(8):
            b.grid[y][x] = EMPTY
    # Row 2: B W W W W W W W  -> No empty at end, so (7,2) not legal
    b.grid[2][0] = BLACK
    for x in range(1, 8):
        b.grid[2][x] = WHITE
    moves = {(m.x, m.y) for m in b.valid_moves(BLACK)}
    assert (7, 2) not in moves

