from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import List, Tuple, Dict

# Reuse constants from main when imported there; mirror here for standalone usage
BOARD_SIZE = 8
EMPTY, BLACK, WHITE = 0, 1, 2


def opponent(player: int) -> int:
    return BLACK if player == WHITE else WHITE


@dataclass
class Params:
    # 4x4 quadrant positional weights flattened row-major (len=16)
    pos: List[float]
    # scalar weights
    mobility_w: float
    stable_w: float

    @staticmethod
    def zeros() -> "Params":
        return Params(pos=[0.0] * 16, mobility_w=0.0, stable_w=0.0)

    def to_dict(self) -> Dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: Dict) -> "Params":
        return Params(pos=list(d["pos"]), mobility_w=float(d["mobility_w"]), stable_w=float(d["stable_w"]))


def sym_index(x: int, y: int) -> int:
    # Map (x,y) in 0..7 to quadrant (qx,qy) in 0..3 with symmetry
    qx = x if x <= 3 else 7 - x
    qy = y if y <= 3 else 7 - y
    return qy * 4 + qx


def compute_pos_features(grid: List[List[int]], player: int) -> List[int]:
    # Returns length-16 vector: for each quadrant index, (#my discs - #opp discs) mapped to that cell group
    feats = [0] * 16
    opp = opponent(player)
    for y in range(BOARD_SIZE):
        for x in range(BOARD_SIZE):
            v = grid[y][x]
            if v == EMPTY:
                continue
            idx = sym_index(x, y)
            if v == player:
                feats[idx] += 1
            elif v == opp:
                feats[idx] -= 1
    return feats


def count_edge_stable(grid: List[List[int]], player: int) -> int:
    # Approximate stable discs by edge stability anchored at corners
    stable: set[Tuple[int, int]] = set()

    def scan_edge(cells: List[Tuple[int, int]]):
        if not cells:
            return
        x0, y0 = cells[0]
        c0 = grid[y0][x0]
        # prefix from first corner
        i = 0
        while i < len(cells) and grid[cells[i][1]][cells[i][0]] == c0 and c0 != EMPTY:
            stable.add(cells[i])
            i += 1
        # suffix from other corner
        x1, y1 = cells[-1]
        c1 = grid[y1][x1]
        j = len(cells) - 1
        while j >= 0 and grid[cells[j][1]][cells[j][0]] == c1 and c1 != EMPTY:
            stable.add(cells[j])
            j -= 1

    # top edge, bottom edge
    scan_edge([(x, 0) for x in range(BOARD_SIZE)])
    scan_edge([(x, BOARD_SIZE - 1) for x in range(BOARD_SIZE)])
    # left edge, right edge
    scan_edge([(0, y) for y in range(BOARD_SIZE)])
    scan_edge([(BOARD_SIZE - 1, y) for y in range(BOARD_SIZE)])

    return sum(1 for (x, y) in stable if grid[y][x] == player)


def evaluate_with_params(board, player: int, params: Params) -> float:
    # Linear evaluation: positional (quadrant) + mobility + edge-stable
    pos_feats = compute_pos_features(board.grid, player)
    val = 0.0
    for i in range(16):
        val += params.pos[i] * pos_feats[i]
    # mobility
    my_mob = len(board.valid_moves(player))
    val += params.mobility_w * my_mob
    # edge-stable discs
    val += params.stable_w * count_edge_stable(board.grid, player)
    return val

