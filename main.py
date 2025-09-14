from __future__ import annotations
import sys
from dataclasses import dataclass
from typing import List, Tuple, Optional

pygame = None  # Lazy import to avoid noisy init message during non-GUI usage
HAVE_PYGAME = False
import json
from pathlib import Path
import concurrent.futures as cf
import os
import random

from eval_params import Params, evaluate_with_params, evaluate_black_advantage


# Constants
BOARD_SIZE = 8
EMPTY, BLACK, WHITE = 0, 1, 2

WINDOW_W, WINDOW_H = 640, 720
BOARD_MARGIN = 20
PANEL_H = 100
GRID_W = WINDOW_W - 2 * BOARD_MARGIN
GRID_H = WINDOW_H - PANEL_H - 2 * BOARD_MARGIN
CELL_W = GRID_W // BOARD_SIZE
CELL_H = GRID_H // BOARD_SIZE

BG_COLOR = (22, 30, 25)
BOARD_COLOR = (33, 94, 44)
LINE_COLOR = (16, 54, 24)
BLACK_COLOR = (18, 18, 18)
WHITE_COLOR = (236, 236, 236)
HINT_COLOR = (255, 230, 0)
TEXT_COLOR = (240, 240, 240)
SUBTLE_TEXT = (200, 200, 200)


DIRECTIONS = [
    (-1, -1), (0, -1), (1, -1),
    (-1, 0),           (1, 0),
    (-1, 1),  (0, 1),  (1, 1)
]


def opponent(player: int) -> int:
    return BLACK if player == WHITE else WHITE


def _evaluate_leaf_for_root(board: 'Board', root_player: int, params: Optional[Params]) -> float:
    # If learned parameters provided, use black-fixed zero-sum evaluation
    if params is not None:
        val = evaluate_black_advantage(board, params)
        return val if root_player == BLACK else -val
    # Heuristic fallback (same as MinimaxAI.evaluate fallback)
    b, w = sum(cell == BLACK for row in board.grid for cell in row), sum(
        cell == WHITE for row in board.grid for cell in row
    )
    my = b if root_player == BLACK else w
    opp = w if root_player == BLACK else b
    disc_total = my + opp
    disc_diff = 0.0 if disc_total == 0 else 100.0 * (my - opp) / disc_total
    my_mob = len(board.valid_moves(root_player))
    opp_mob = len(board.valid_moves(opponent(root_player)))
    mob_total = my_mob + opp_mob
    mobility = 0.0 if mob_total == 0 else 100.0 * (my_mob - opp_mob) / mob_total
    my_corners = 0
    opp_corners = 0
    for (x, y) in MinimaxAI.CORNERS:
        cell = board.grid[y][x]
        if cell == root_player:
            my_corners += 1
        elif cell == opponent(root_player):
            opp_corners += 1
    corner_score = 25.0 * (my_corners - opp_corners)
    pos_score = 0
    for y in range(BOARD_SIZE):
        for x in range(BOARD_SIZE):
            v = board.grid[y][x]
            if v == EMPTY:
                continue
            weight = MinimaxAI.POS_WEIGHTS[y][x]
            if v == root_player:
                pos_score += weight
            elif v == opponent(root_player):
                pos_score -= weight
    return 0.6 * pos_score + 0.2 * corner_score + 0.15 * mobility + 0.05 * disc_diff


def _search_value_standalone(
    board: 'Board', to_move: int, depth: int, alpha: float, beta: float, root_player: int, params: Optional[Params]
) -> float:
    import math
    my_moves = board.valid_moves(to_move)
    opp_to_move = opponent(to_move)
    opp_moves = board.valid_moves(opp_to_move)
    if depth <= 0 or board.full() or (not my_moves and not opp_moves):
        return _evaluate_leaf_for_root(board, root_player, params)
    if not my_moves:
        return _search_value_standalone(board, opp_to_move, depth - 1, alpha, beta, root_player, params)
    if to_move == root_player:
        value = -math.inf
        for m in my_moves:
            nb = board.copy()
            nb.apply_move(m, to_move)
            value = max(value, _search_value_standalone(nb, opp_to_move, depth - 1, alpha, beta, root_player, params))
            alpha = max(alpha, value)
            if alpha >= beta:
                break
        return value
    else:
        value = math.inf
        for m in my_moves:
            nb = board.copy()
            nb.apply_move(m, to_move)
            value = min(value, _search_value_standalone(nb, opp_to_move, depth - 1, alpha, beta, root_player, params))
            beta = min(beta, value)
            if beta <= alpha:
                break
        return value


@dataclass
class Move:
    x: int
    y: int
    flips: List[Tuple[int, int]]


class Board:
    def __init__(self):
        self.size = BOARD_SIZE
        self.grid: List[List[int]] = [[EMPTY for _ in range(self.size)] for _ in range(self.size)]
        mid = self.size // 2
        self.grid[mid - 1][mid - 1] = WHITE
        self.grid[mid][mid] = WHITE
        self.grid[mid - 1][mid] = BLACK
        self.grid[mid][mid - 1] = BLACK

    def copy(self) -> 'Board':
        nb = Board()
        nb.grid = [row.copy() for row in self.grid]
        return nb

    def inside(self, x: int, y: int) -> bool:
        return 0 <= x < self.size and 0 <= y < self.size

    def valid_moves(self, player: int) -> List[Move]:
        moves: List[Move] = []
        for y in range(self.size):
            for x in range(self.size):
                if self.grid[y][x] != EMPTY:
                    continue
                flips: List[Tuple[int, int]] = []
                for dx, dy in DIRECTIONS:
                    path: List[Tuple[int, int]] = []
                    cx, cy = x + dx, y + dy
                    while self.inside(cx, cy) and self.grid[cy][cx] == opponent(player):
                        path.append((cx, cy))
                        cx += dx
                        cy += dy
                    if path and self.inside(cx, cy) and self.grid[cy][cx] == player:
                        flips.extend(path)
                if flips:
                    moves.append(Move(x, y, flips))
        return moves

    def apply_move(self, move: Move, player: int) -> None:
        self.grid[move.y][move.x] = player
        for fx, fy in move.flips:
            self.grid[fy][fx] = player

    def count(self) -> Tuple[int, int]:
        b = sum(cell == BLACK for row in self.grid for cell in row)
        w = sum(cell == WHITE for row in self.grid for cell in row)
        return b, w

    def full(self) -> bool:
        return all(cell != EMPTY for row in self.grid for cell in row)


class SimpleAI:
    """A small heuristic AI: prioritize corners > max flips."""

    CORNERS = {(0, 0), (0, BOARD_SIZE - 1), (BOARD_SIZE - 1, 0), (BOARD_SIZE - 1, BOARD_SIZE - 1)}

    def choose(self, board: Board, player: int) -> Optional[Move]:
        moves = board.valid_moves(player)
        if not moves:
            return None
        # Favor corners
        for m in moves:
            if (m.x, m.y) in self.CORNERS:
                return m
        # Otherwise, pick move that flips the most
        return max(moves, key=lambda m: len(m.flips))


class MinimaxAI:
    """Minimax with alpha-beta pruning and a positional evaluation.

    Depth is measured in plies (half-moves).
    """

    POS_WEIGHTS = [
        [100, -20, 10, 5, 5, 10, -20, 100],
        [-20, -50, -2, -2, -2, -2, -50, -20],
        [10, -2, 5, 1, 1, 5, -2, 10],
        [5, -2, 1, 0, 0, 1, -2, 5],
        [5, -2, 1, 0, 0, 1, -2, 5],
        [10, -2, 5, 1, 1, 5, -2, 10],
        [-20, -50, -2, -2, -2, -2, -50, -20],
        [100, -20, 10, 5, 5, 10, -20, 100],
    ]

    CORNERS = {(0, 0), (0, BOARD_SIZE - 1), (BOARD_SIZE - 1, 0), (BOARD_SIZE - 1, BOARD_SIZE - 1)}

    def __init__(self, depth: int = 3, params: Optional[Params] = None, workers: int = 1):
        self.depth = depth
        self.params = params
        self.workers = max(1, int(workers))

    def choose(self, board: 'Board', player: int) -> Optional[Move]:
        import math
        best_score = -math.inf
        best_move: Optional[Move] = None
        moves = board.valid_moves(player)
        if not moves:
            return None
        # Shuffle move order to avoid deterministic tie-bias (seeded upstream via random.seed)
        random.shuffle(moves)

        # simple move ordering: corners first, then by flips desc
        def key_fn(m: Move):
            return ((m.x, m.y) in self.CORNERS, len(m.flips))
        moves.sort(key=key_fn, reverse=True)

        # Parallelize root moves if requested
        if self.workers > 1 and len(moves) > 1 and self.depth > 1:
            args_list = []
            for m in moves:
                nb = board.copy()
                nb.apply_move(m, player)
                args_list.append((nb, opponent(player), self.depth - 1, -math.inf, math.inf, player, self.params))
            with cf.ProcessPoolExecutor(max_workers=self.workers) as ex:
                futures = {ex.submit(_search_value_standalone, *args): m for args, m in zip(args_list, moves)}
                for fut in cf.as_completed(futures):
                    m = futures[fut]
                    try:
                        score = fut.result()
                    except Exception:
                        continue
                    if score > best_score:
                        best_score = score
                        best_move = m
        else:
            for m in moves:
                nb = board.copy()
                nb.apply_move(m, player)
                score = self._search(nb, opponent(player), self.depth - 1, -math.inf, math.inf, player)
                if score > best_score:
                    best_score = score
                    best_move = m
        return best_move

    def _search(self, board: 'Board', to_move: int, depth: int, alpha: float, beta: float, root_player: int) -> float:
        import math
        # Terminal or depth limit
        my_moves = board.valid_moves(to_move)
        opp_moves = board.valid_moves(opponent(to_move))
        if depth <= 0 or board.full() or (not my_moves and not opp_moves):
            return self.evaluate(board, root_player)

        # Handle pass
        if not my_moves:
            # reduce depth on pass to ensure progress
            return self._search(board, opponent(to_move), depth - 1, alpha, beta, root_player)

        # Move ordering: corners first, then flips
        def key_fn(m: Move):
            return ((m.x, m.y) in self.CORNERS, len(m.flips))
        ordered = sorted(my_moves, key=key_fn, reverse=True)

        if to_move == root_player:
            value = -math.inf
            for m in ordered:
                nb = board.copy()
                nb.apply_move(m, to_move)
                value = max(value, self._search(nb, opponent(to_move), depth - 1, alpha, beta, root_player))
                alpha = max(alpha, value)
                if alpha >= beta:
                    break
            return value
        else:
            value = math.inf
            for m in ordered:
                nb = board.copy()
                nb.apply_move(m, to_move)
                value = min(value, self._search(nb, opponent(to_move), depth - 1, alpha, beta, root_player))
                beta = min(beta, value)
                if beta <= alpha:
                    break
            return value

    def evaluate(self, board: 'Board', player: int) -> float:
        # If learned parameters provided, use zero-sum black-fixed evaluation
        if self.params is not None:
            val = evaluate_black_advantage(board, self.params)
            return val if player == BLACK else -val
        # Heuristic fallback
        b, w = board.count()
        my = b if player == BLACK else w
        opp = w if player == BLACK else b
        disc_total = my + opp
        disc_diff = 0.0 if disc_total == 0 else 100.0 * (my - opp) / disc_total
        my_mob = len(board.valid_moves(player))
        opp_mob = len(board.valid_moves(opponent(player)))
        mob_total = my_mob + opp_mob
        mobility = 0.0 if mob_total == 0 else 100.0 * (my_mob - opp_mob) / mob_total
        my_corners = 0
        opp_corners = 0
        for (x, y) in self.CORNERS:
            cell = board.grid[y][x]
            if cell == player:
                my_corners += 1
            elif cell == opponent(player):
                opp_corners += 1
        corner_score = 25.0 * (my_corners - opp_corners)
        pos_score = 0
        for y in range(BOARD_SIZE):
            for x in range(BOARD_SIZE):
                v = board.grid[y][x]
                if v == EMPTY:
                    continue
                weight = self.POS_WEIGHTS[y][x]
                if v == player:
                    pos_score += weight
                elif v == opponent(player):
                    pos_score -= weight
        return 0.6 * pos_score + 0.2 * corner_score + 0.15 * mobility + 0.05 * disc_diff


class Game:
    def __init__(self):
        global pygame, HAVE_PYGAME
        if not HAVE_PYGAME:
            try:
                import pygame as _pygame  # type: ignore
                pygame = _pygame
                HAVE_PYGAME = True
            except Exception:
                raise ImportError("pygame is required to run the GUI. Install with 'pip install pygame'.")
        pygame.init()
        pygame.display.set_caption("Reversi (Pygame)")
        self.screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Arial", 22)
        self.big_font = pygame.font.SysFont("Arial", 28, bold=True)

        self.board = Board()
        self.current = BLACK
        self.show_hints = True
        self.ai_enabled = True  # White plays by AI by default
        # Load learned parameters if available
        params: Optional[Params] = None
        # Prefer versioned latest first
        latest = Path("weights") / "latest.json"
        candidates = [latest, Path("weights.json")]
        # Also consider newest file in weights/ if latest missing
        weights_dir = Path("weights")
        if not latest.exists() and weights_dir.exists():
            try:
                newest = max((p for p in weights_dir.glob("*.json")), key=lambda p: p.stat().st_mtime, default=None)
            except Exception:
                newest = None
            if newest:
                candidates.insert(0, newest)
        for p in candidates:
            if p.exists():
                try:
                    data = json.loads(p.read_text())
                    params = Params.from_dict(data)
                    self.message = f"Loaded learned params from {p.name}. Black to move"
                    try:
                        print(f"[weights] Loaded for game: {p}", flush=True)
                    except Exception:
                        pass
                    break
                except Exception:
                    continue
        self.loaded_params: Optional[Params] = params
        self.use_learned: bool = params is not None
        self.depth: int = 4  # default deeper search
        workers_env = os.getenv("MINIMAX_WORKERS", "1")
        try:
            workers = int(workers_env)
        except Exception:
            workers = 1
        self.ai = MinimaxAI(depth=self.depth, params=(self.loaded_params if self.use_learned else None), workers=workers)
        self.game_over = False
        self.message = "Black to move"
        if not self.use_learned:
            try:
                print("[weights] No learned weights found. Using heuristic evaluator.", flush=True)
            except Exception:
                pass

    def reset(self):
        self.board = Board()
        self.current = BLACK
        self.game_over = False
        self.message = "Black to move"
        # keep toggles

    def cell_rect(self, x: int, y: int) -> pygame.Rect:
        left = BOARD_MARGIN + x * CELL_W
        top = BOARD_MARGIN + y * CELL_H
        return pygame.Rect(left, top, CELL_W, CELL_H)

    def draw_board(self):
        # background
        self.screen.fill(BG_COLOR)

        # board area
        board_rect = pygame.Rect(BOARD_MARGIN, BOARD_MARGIN, CELL_W * BOARD_SIZE, CELL_H * BOARD_SIZE)
        pygame.draw.rect(self.screen, BOARD_COLOR, board_rect, border_radius=8)

        # grid lines
        for i in range(BOARD_SIZE + 1):
            x = BOARD_MARGIN + i * CELL_W
            y = BOARD_MARGIN + i * CELL_H
            pygame.draw.line(self.screen, LINE_COLOR, (x, BOARD_MARGIN), (x, BOARD_MARGIN + CELL_H * BOARD_SIZE), 2)
            pygame.draw.line(self.screen, LINE_COLOR, (BOARD_MARGIN, y), (BOARD_MARGIN + CELL_W * BOARD_SIZE, y), 2)

        # discs
        for y in range(BOARD_SIZE):
            for x in range(BOARD_SIZE):
                cell = self.board.grid[y][x]
                if cell == EMPTY:
                    continue
                cx = BOARD_MARGIN + x * CELL_W + CELL_W // 2
                cy = BOARD_MARGIN + y * CELL_H + CELL_H // 2
                r = min(CELL_W, CELL_H) // 2 - 6
                color = BLACK_COLOR if cell == BLACK else WHITE_COLOR
                pygame.draw.circle(self.screen, color, (cx, cy), r)

        # hints
        if self.show_hints and not self.game_over:
            for m in self.board.valid_moves(self.current):
                rect = self.cell_rect(m.x, m.y)
                pad = 6
                hint_rect = pygame.Rect(rect.left + pad, rect.top + pad, rect.width - 2 * pad, rect.height - 2 * pad)
                pygame.draw.rect(self.screen, HINT_COLOR, hint_rect, width=2, border_radius=6)

        # panel (wrapped text to avoid overflow)
        panel_top = BOARD_MARGIN + CELL_H * BOARD_SIZE + 10
        b_count, w_count = self.board.count()
        turn_text = f"Turn: {'Black' if self.current == BLACK else 'White'}"
        score_text = f"Score: Black {b_count} : {w_count} White"
        eval_mode = 'Learned' if (self.use_learned and self.loaded_params is not None) else 'Heuristic'
        controls_text = (
            f"Depth:{self.depth}  Eval:{eval_mode}   [H]ints  [A]I  [L]earned  [ [ / ] ] depth  [R]estart"
        )

        max_w = CELL_W * BOARD_SIZE
        x = BOARD_MARGIN
        y = panel_top
        y = self._blit_wrapped(self.message, self.big_font, TEXT_COLOR, x, y, max_w)
        y = self._blit_wrapped(f"{turn_text}   {score_text}", self.font, SUBTLE_TEXT, x, y + 4, max_w)
        _ = self._blit_wrapped(controls_text, self.font, SUBTLE_TEXT, x, y + 2, max_w)

    def _blit_wrapped(self, text: str, font, color, x: int, y: int, max_width: int) -> int:
        """Render text with simple word-wrapping. Returns next y position."""
        words = text.split()
        if not words:
            return y
        line = ""
        cur_y = y
        for word in words:
            test = word if not line else f"{line} {word}"
            w, h = font.size(test)
            if w <= max_width:
                line = test
            else:
                surf = font.render(line, True, color)
                self.screen.blit(surf, (x, cur_y))
                cur_y += h + 2
                line = word
        if line:
            surf = font.render(line, True, color)
            self.screen.blit(surf, (x, cur_y))
            cur_y += surf.get_height()
        return cur_y

    def pos_to_cell(self, pos: Tuple[int, int]) -> Optional[Tuple[int, int]]:
        x, y = pos
        if not (BOARD_MARGIN <= x < BOARD_MARGIN + CELL_W * BOARD_SIZE):
            return None
        if not (BOARD_MARGIN <= y < BOARD_MARGIN + CELL_H * BOARD_SIZE):
            return None
        cx = (x - BOARD_MARGIN) // CELL_W
        cy = (y - BOARD_MARGIN) // CELL_H
        return int(cx), int(cy)

    def handle_click(self, pos: Tuple[int, int]):
        if self.game_over:
            return
        cell = self.pos_to_cell(pos)
        if cell is None:
            return
        x, y = cell
        # perform move if valid
        candidates = {(m.x, m.y): m for m in self.board.valid_moves(self.current)}
        move = candidates.get((x, y))
        if move:
            self.board.apply_move(move, self.current)
            self.after_move()
        else:
            self.message = "Invalid move"

    def rebuild_ai(self):
        # Recreate Minimax with current depth and eval mode
        self.ai = MinimaxAI(depth=self.depth, params=(self.loaded_params if self.use_learned else None))

    def after_move(self):
        # switch turn or pass if needed, check game over
        self.current = opponent(self.current)
        moves = self.board.valid_moves(self.current)
        if not moves:
            # pass
            self.current = opponent(self.current)
            if not self.board.valid_moves(self.current):
                self.finish_game()
            else:
                self.message = f"{'Black' if self.current == BLACK else 'White'} (opponent) had no moves — your turn again"
        else:
            self.message = f"{'Black' if self.current == BLACK else 'White'} to move"

    def finish_game(self):
        self.game_over = True
        b, w = self.board.count()
        if b > w:
            self.message = f"Game over — Black wins {b}:{w}"
        elif w > b:
            self.message = f"Game over — White wins {w}:{b}"
        else:
            self.message = f"Game over — Draw {b}:{w}"

    def maybe_ai_move(self):
        if self.game_over:
            return
        if self.ai_enabled and self.current == WHITE:
            move = self.ai.choose(self.board, WHITE)
            if move:
                self.board.apply_move(move, WHITE)
                self.after_move()
            else:
                # pass
                self.current = BLACK
                if not self.board.valid_moves(self.current):
                    self.finish_game()
                else:
                    self.message = "White passed — Black to move"

    def run(self):
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit(0)
                elif event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_ESCAPE, pygame.K_q):
                        pygame.quit()
                        sys.exit(0)
                    elif event.key == pygame.K_r:
                        self.reset()
                    elif event.key == pygame.K_h:
                        self.show_hints = not self.show_hints
                    elif event.key == pygame.K_a:
                        self.ai_enabled = not self.ai_enabled
                    elif event.key == pygame.K_l:
                        # toggle learned/heuristic evaluation
                        self.use_learned = not self.use_learned and (self.loaded_params is not None)
                        self.rebuild_ai()
                    elif event.key == pygame.K_LEFTBRACKET:
                        # decrease depth
                        if self.depth > 1:
                            self.depth -= 1
                            self.rebuild_ai()
                    elif event.key == pygame.K_RIGHTBRACKET:
                        # increase depth (cap to avoid too slow)
                        if self.depth < 8:
                            self.depth += 1
                            self.rebuild_ai()
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    # Only let human act on Black by default
                    if self.current == BLACK or not self.ai_enabled:
                        self.handle_click(event.pos)

            # AI move if enabled
            self.maybe_ai_move()

            # draw
            self.draw_board()
            pygame.display.flip()
            self.clock.tick(60)


if __name__ == "__main__":
    Game().run()
