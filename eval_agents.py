from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
import concurrent.futures as cf
from pathlib import Path
from typing import Optional, Tuple, List, Iterable

from eval_params import Params
from main import Board, BLACK, WHITE, MinimaxAI


def load_params_from_candidates(explicit: Optional[Path] = None) -> Optional[Params]:
    if explicit is not None:
        p = explicit
        if p.is_dir():
            # use latest in dir
            cands = sorted(p.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True)
            if not cands:
                return None
            p = cands[0]
        try:
            params = Params.from_dict(json.loads(p.read_text()))
            try:
                print(f"[weights] Loaded: {p}", flush=True)
            except Exception:
                pass
            return params
        except Exception:
            return None
    # fallback to standard candidates
    weights_dir = Path("weights")
    candidates = [weights_dir / "latest.json", Path("weights.json")]
    if not candidates[0].exists() and weights_dir.exists():
        try:
            newest = max((q for q in weights_dir.glob("*.json")), key=lambda q: q.stat().st_mtime, default=None)
        except Exception:
            newest = None
        if newest:
            candidates.insert(0, newest)
    for p in candidates:
        if p.exists():
            try:
                params = Params.from_dict(json.loads(p.read_text()))
                try:
                    print(f"[weights] Loaded: {p}", flush=True)
                except Exception:
                    pass
                return params
            except Exception:
                continue
    return None


def load_params_pool_entries(
    dir_path: Path,
    exclude: Optional[Path] = None,
    order: str = "newest",
) -> List[Tuple[Path, Params]]:
    """Return an ordered list of (path, params) for all JSON weights in dir.

    order: 'newest' (mtime desc), 'oldest' (mtime asc), 'name' (filename asc)
    """
    entries: List[Tuple[Path, Params]] = []
    if not dir_path.exists() or not dir_path.is_dir():
        return entries
    files = [p for p in dir_path.glob("*.json") if p.is_file()]
    if exclude is not None:
        files = [p for p in files if p.resolve() != exclude.resolve()]
    if order == "newest":
        files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    elif order == "oldest":
        files.sort(key=lambda p: p.stat().st_mtime)
    elif order == "name":
        files.sort(key=lambda p: p.name)
    else:
        raise ValueError("order must be 'newest', 'oldest', or 'name'")
    for p in files:
        try:
            entries.append((p, Params.from_dict(json.loads(p.read_text()))))
        except Exception:
            continue
    return entries


def load_params_from_paths(paths: List[Path]) -> List[Tuple[Path, Params]]:
    entries: List[Tuple[Path, Params]] = []
    for p in paths:
        try:
            entries.append((p, Params.from_dict(json.loads(p.read_text()))))
        except Exception:
            continue
    return entries


class RandomAI:
    def choose(self, board: Board, player: int):
        moves = board.valid_moves(player)
        if not moves:
            return None
        return random.choice(moves)


class EpsilonPolicy:
    """Decorator to add epsilon randomness on top of a base AI."""

    def __init__(self, base_ai, epsilon: float):
        self.base_ai = base_ai
        try:
            self.epsilon = max(0.0, float(epsilon))
        except Exception:
            self.epsilon = 0.0

    def choose(self, board: Board, player: int):
        moves = board.valid_moves(player)
        if not moves:
            return None
        if self.epsilon > 0.0 and random.random() < self.epsilon:
            return random.choice(moves)
        return self.base_ai.choose(board, player)


def play_game(black_ai, white_ai, random_openings: int = 0, seed: Optional[int] = None) -> Tuple[int, int, int]:
    if seed is not None:
        random.seed(seed)
    b = Board()
    player = BLACK
    terminal = False

    # random opening plies to diversify
    ro = random_openings
    while ro > 0 and not terminal:
        moves = b.valid_moves(player)
        if not moves:
            player = WHITE if player == BLACK else BLACK
            if not b.valid_moves(player):
                terminal = True
            ro -= 1
            continue
        m = random.choice(moves)
        b.apply_move(m, player)
        player = WHITE if player == BLACK else BLACK
        ro -= 1

    while not terminal:
        ai = black_ai if player == BLACK else white_ai
        move = ai.choose(b, player)
        if move is not None:
            b.apply_move(move, player)
            player = WHITE if player == BLACK else BLACK
        else:
            player = WHITE if player == BLACK else BLACK
            if not b.valid_moves(player):
                terminal = True

    bcount, wcount = b.count()
    if bcount > wcount:
        return (1, 0, 0)
    elif wcount > bcount:
        return (0, 1, 0)
    else:
        return (0, 0, 1)


def _corner_audit_one(spec: dict) -> Tuple[int, int]:
    depth = spec["depth"]
    our_params = spec["our_params"]
    opp_kind = spec["opp_kind"]
    opp_params = spec.get("opp_params")
    our_is_black = spec["our_is_black"]
    random_openings = spec["random_openings"]
    seed = spec.get("seed")

    if seed is not None:
        random.seed(seed)

    epsilon = float(spec.get("epsilon_move", 0.0) or 0.0)
    our_ai = MinimaxAI(depth=depth, params=our_params, workers=1)
    if opp_kind == "heuristic":
        opp_ai = MinimaxAI(depth=depth, params=None, workers=1)
    elif opp_kind == "random":
        opp_ai = RandomAI()
    else:
        opp_ai = MinimaxAI(depth=depth, params=opp_params, workers=1)

    if epsilon > 0.0:
        our_ai = EpsilonPolicy(our_ai, epsilon)
        if not isinstance(opp_ai, RandomAI):
            opp_ai = EpsilonPolicy(opp_ai, epsilon)

    black_ai = our_ai if our_is_black else opp_ai
    white_ai = opp_ai if our_is_black else our_ai

    b = Board()
    player = BLACK
    terminal = False
    ro = random_openings
    while ro > 0 and not terminal:
        moves = b.valid_moves(player)
        if not moves:
            player = WHITE if player == BLACK else BLACK
            if not b.valid_moves(player):
                terminal = True
            ro -= 1
            continue
        m = random.choice(moves)
        b.apply_move(m, player)
        player = WHITE if player == BLACK else BLACK
        ro -= 1

    CORNERS = {(0, 0), (0, 7), (7, 0), (7, 7)}
    opportunities = 0
    taken = 0
    while not terminal:
        ai = black_ai if player == BLACK else white_ai
        is_ours = (our_is_black and player == BLACK) or ((not our_is_black) and player == WHITE)
        legal = b.valid_moves(player)
        if is_ours and legal:
            coords = {(m.x, m.y) for m in legal}
            corner_moves = CORNERS & coords
            if corner_moves:
                opportunities += 1
                m_choice = ai.choose(b, player)
                if m_choice and (m_choice.x, m_choice.y) in CORNERS:
                    taken += 1
                if m_choice:
                    b.apply_move(m_choice, player)
                    player = WHITE if player == BLACK else BLACK
                    continue
        move = ai.choose(b, player)
        if move is not None:
            b.apply_move(move, player)
            player = WHITE if player == BLACK else BLACK
        else:
            player = WHITE if player == BLACK else BLACK
            if not b.valid_moves(player):
                terminal = True

    return opportunities, taken


def corner_audit(
    params: Optional[Params],
    opponent_kind: str = "heuristic",
    cfg: Optional['EvalConfig'] = None,
    opp_dir: Optional[Path] = None,
    opp_order: str = "newest",
    opp_limit: int = 0,
    matches_per_opp: int = 0,
    progress: bool = False,
    workers: int = 1,
) -> dict:
    """Measure how often our agent takes a corner when available.

    Returns: { opportunities, taken, rate, games }
    """
    # Prepare our agent
    cfg = cfg or EvalConfig()
    our_ai = MinimaxAI(depth=cfg.depth, params=params)

    # Prepare opponents deterministically
    if opponent_kind not in {"heuristic", "random", "pool"}:
        raise ValueError("opponent_kind must be 'heuristic', 'random', or 'pool'")

    pool_entries: List[Tuple[Path, Params]] = []
    if opponent_kind == "heuristic":
        opp_ai = MinimaxAI(depth=cfg.depth, params=None)
    elif opponent_kind == "random":
        opp_ai = RandomAI()
    else:
        d = opp_dir if opp_dir is not None else Path("weights")
        exclude_path = None
        if d.exists():
            latest = d / "latest.json"
            if latest.exists():
                exclude_path = latest
        pool_entries = load_params_pool_entries(d, exclude=exclude_path, order=opp_order)
        if opp_limit and opp_limit > 0:
            pool_entries = pool_entries[:opp_limit]
        if not pool_entries:
            opp_ai = MinimaxAI(depth=cfg.depth, params=None)
            opponent_kind = "heuristic"
        else:
            opp_ai = None  # set per game

    total = cfg.matches
    # Schedule opponents
    if opponent_kind == "pool" and pool_entries:
        # If scheduling per opponent, deduplicate opponents by file name to avoid
        # accidentally scheduling multiples of what is effectively the same opponent.
        if matches_per_opp and matches_per_opp > 0:
            seen_names = set()
            pool_entries = [e for e in pool_entries if (e[0].name not in seen_names and not seen_names.add(e[0].name))]
            schedule: List[int] = []
            for idx in range(len(pool_entries)):
                schedule.extend([idx] * matches_per_opp)
            total = len(schedule)
        else:
            schedule = [i % len(pool_entries) for i in range(total)]
    else:
        schedule = [-1 for _ in range(total)]

    CORNERS = {(0, 0), (0, 7), (7, 0), (7, 7)}
    opportunities = 0
    taken = 0

    specs: List[dict] = []
    labels: List[str] = []
    for i in range(total):
        our_is_black = not (cfg.swap_colors and (i % 2 == 1))
        seed = None if cfg.seed is None else cfg.seed + i
        if opponent_kind == "pool" and pool_entries:
            chosen_idx = schedule[i]
            chosen_path, chosen_params = pool_entries[chosen_idx]
            labels.append(chosen_path.name)
            specs.append(
                {
                    "depth": cfg.depth,
                    "our_params": params,
                    "opp_kind": "pool",
                    "opp_params": chosen_params,
                    "our_is_black": our_is_black,
                    "random_openings": cfg.random_openings,
                    "seed": seed,
                }
            )
        else:
            labels.append(opponent_kind)
            specs.append(
                {
                    "depth": cfg.depth,
                    "our_params": params,
                    "opp_kind": opponent_kind,
                    "opp_params": None,
                    "our_is_black": our_is_black,
                    "random_openings": cfg.random_openings,
                    "seed": seed,
                }
            )

    if workers and workers > 1:
        with cf.ProcessPoolExecutor(max_workers=workers) as ex:
            futures = {ex.submit(_corner_audit_one, spec): i for i, spec in enumerate(specs)}
            for fut in cf.as_completed(futures):
                i = futures[fut]
                opps, tk = fut.result()
                opportunities += opps
                taken += tk
                if progress:
                    rate = (taken / opportunities) if opportunities else 0.0
                    print(
                        f"[corner] game {i+1}/{total} vs {labels[i]}  opportunities:{opportunities} taken:{taken} rate:{rate:.1%}",
                        flush=True,
                    )
    else:
        for i, spec in enumerate(specs):
            opps, tk = _corner_audit_one(spec)
            opportunities += opps
            taken += tk
            if progress:
                rate = (taken / opportunities) if opportunities else 0.0
                print(
                    f"[corner] game {i+1}/{total} vs {labels[i]}  opportunities:{opportunities} taken:{taken} rate:{rate:.1%}",
                    flush=True,
                )

    rate = (taken / opportunities) if opportunities else 0.0
    return {"opportunities": opportunities, "taken": taken, "rate": rate, "games": total}


@dataclass
class EvalConfig:
    matches: int = 50
    depth: int = 4
    random_openings: int = 4
    swap_colors: bool = True
    seed: Optional[int] = None


def _eval_one_game(spec: dict) -> Tuple[int, int, int]:
    # Child-safe single game evaluation, returns (wins, losses, draws) from our POV
    depth = spec["depth"]
    our_params = spec["our_params"]
    opp_kind = spec["opp_kind"]
    opp_params = spec.get("opp_params")
    our_is_black = spec["our_is_black"]
    random_openings = spec["random_openings"]
    seed = spec.get("seed")

    epsilon = float(spec.get("epsilon_move", 0.0) or 0.0)
    our_ai = MinimaxAI(depth=depth, params=our_params, workers=1)
    if opp_kind == "heuristic":
        opp_ai = MinimaxAI(depth=depth, params=None, workers=1)
    elif opp_kind == "random":
        opp_ai = RandomAI()
    else:
        opp_ai = MinimaxAI(depth=depth, params=opp_params, workers=1)

    if epsilon > 0.0:
        our_ai = EpsilonPolicy(our_ai, epsilon)
        if not isinstance(opp_ai, RandomAI):
            opp_ai = EpsilonPolicy(opp_ai, epsilon)

    if our_is_black:
        b_ai, w_ai = (our_ai, opp_ai)
    else:
        b_ai, w_ai = (opp_ai, our_ai)
    w, l, d = play_game(b_ai, w_ai, random_openings=random_openings, seed=seed)
    if our_is_black:
        return (w, l, d)
    else:
        return (l, w, d)


def evaluate(
    params: Optional[Params],
    opponent_kind: str = "heuristic",
    cfg: EvalConfig = EvalConfig(),
    progress: bool = False,
    opp_dir: Optional[Path] = None,
    opp_order: str = "newest",
    opp_limit: int = 0,
    matches_per_opp: int = 0,
    workers: int = 1,
    opp_paths: Optional[List[Path]] = None,
    opp_skip_newest: int = 0,
    epsilon_move: float = 0.0,
    parallel: str = "process",
) -> dict:
    # Prepare agents
    if opponent_kind not in {"heuristic", "random", "pool"}:
        raise ValueError("opponent_kind must be 'heuristic', 'random', or 'pool'")

    black_first = MinimaxAI(depth=cfg.depth, params=params)
    pool_entries: List[Tuple[Path, Params]] = []
    if opponent_kind == "heuristic":
        opp_ai = MinimaxAI(depth=cfg.depth, params=None)
    elif opponent_kind == "random":
        opp_ai = RandomAI()
    else:
        # pool
        if opp_paths:
            pool_entries = load_params_from_paths(opp_paths)
        else:
            d = opp_dir if opp_dir is not None else Path("weights")
            # Try to exclude latest if current params loaded from there
            exclude_path = None
            if d.exists():
                latest = d / "latest.json"
                if latest.exists():
                    exclude_path = latest
            pool_entries = load_params_pool_entries(d, exclude=exclude_path, order=opp_order)
            if opp_skip_newest and opp_skip_newest > 0:
                pool_entries = pool_entries[opp_skip_newest:]
            if opp_limit and opp_limit > 0:
                pool_entries = pool_entries[:opp_limit]
        if not pool_entries:
            print("[eval] Opponent pool is empty. Falling back to heuristic.")
            opp_ai = MinimaxAI(depth=cfg.depth, params=None)
            opponent_kind = "heuristic"
        else:
            # Placeholder; will pick each game
            opp_ai = None  # type: ignore

    total = cfg.matches
    wins = losses = draws = 0
    # Prepare schedule of opponents (deterministic)
    if opponent_kind == "pool" and pool_entries:
        if matches_per_opp and matches_per_opp > 0:
            schedule: List[int] = []
            for idx in range(len(pool_entries)):
                schedule.extend([idx] * matches_per_opp)
            total = len(schedule)
        else:
            # round-robin across pool to fill total matches
            schedule = [i % len(pool_entries) for i in range(total)]
    else:
        schedule = [-1 for _ in range(total)]  # -1 means fixed opp_ai

    # Build job specs
    specs: List[dict] = []
    labels: List[str] = []
    for i in range(total):
        our_is_black = not (cfg.swap_colors and (i % 2 == 1))
        seed = None if cfg.seed is None else cfg.seed + i
        if opponent_kind == "pool" and pool_entries:
            chosen_idx = schedule[i]
            chosen_path, chosen_params = pool_entries[chosen_idx]
            labels.append(chosen_path.name)
            spec = {
                "depth": cfg.depth,
                "our_params": params,
                "opp_kind": "pool",
                "opp_params": chosen_params,
                "our_is_black": our_is_black,
                "random_openings": cfg.random_openings,
                "seed": seed,
                "epsilon_move": epsilon_move,
            }
        else:
            labels.append(opponent_kind)
            spec = {
                "depth": cfg.depth,
                "our_params": params,
                "opp_kind": opponent_kind,
                "opp_params": None,
                "our_is_black": our_is_black,
                "random_openings": cfg.random_openings,
                "seed": seed,
                "epsilon_move": epsilon_move,
            }
        specs.append(spec)

    # Print an informative start summary with actual totals/opponents
    if progress:
        # Compute actual opponents used in the schedule (after skip/limit and matches_per_opp expansion)
        def uniq_in_order(seq: List[str]) -> List[str]:
            seen = set()
            out: List[str] = []
            for s in seq:
                if s not in seen:
                    seen.add(s)
                    out.append(s)
            return out
        opp_names = uniq_in_order(labels) if (opponent_kind == "pool" and labels) else []
        opp_count = len(opp_names) if opp_names else 1
        mode = "matches_per_opp" if (opponent_kind == "pool" and matches_per_opp and matches_per_opp > 0) else "fixed_matches"
        # Clear, explicit schedule summary
        if mode == "matches_per_opp":
            print(
                f"[eval] Plan: matches_per_opp={matches_per_opp} x opponents={opp_count} => total_games={total} (depth={cfg.depth}, rand_openings={cfg.random_openings})",
                flush=True,
            )
        else:
            print(
                f"[eval] Plan: total_games={total} (requested matches={cfg.matches}) depth={cfg.depth} rand_openings={cfg.random_openings}",
                flush=True,
            )
        if opp_names:
            preview = ", ".join(opp_names[:4]) + (" ..." if len(opp_names) > 4 else "")
            print(f"[eval] Opponents: {preview}", flush=True)

    if workers and workers > 1:
        # Parallelize at the game level. For CPU-bound Minimax, process pool is typically faster.
        Executor = cf.ProcessPoolExecutor if parallel == "process" else cf.ThreadPoolExecutor
        with Executor(max_workers=workers) as ex:
            futures = {ex.submit(_eval_one_game, specs[i]): i for i in range(len(specs))}
            played = 0
            for fut in cf.as_completed(futures):
                i = futures[fut]
                w, l, d = fut.result()
                wins += w
                losses += l
                draws += d
                played += 1
                if progress:
                    wr = wins / played if played else 0.0
                    pov = "(our=Black)" if specs[i]["our_is_black"] else "(our=White)"
                    print(
                        f"[eval] {played}/{total} vs {labels[i]} {pov}  W:{wins} L:{losses} D:{draws}  win_rate:{wr:.1%}",
                        flush=True,
                    )
    else:
        for i, spec in enumerate(specs):
            w, l, d = _eval_one_game(spec)
            wins += w
            losses += l
            draws += d
            if progress:
                played = i + 1
                wr = wins / played if played else 0.0
                pov = "(our=Black)" if spec["our_is_black"] else "(our=White)"
                print(
                    f"[eval] {played}/{total} vs {labels[i]} {pov}  W:{wins} L:{losses} D:{draws}  win_rate:{wr:.1%}",
                    flush=True,
                )

    games = wins + losses + draws
    win_rate = wins / games if games else 0.0
    return {
        "games": games,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "win_rate": win_rate,
    }


def main():
    ap = argparse.ArgumentParser(description="Evaluate weights vs heuristic/random/pool opponent")
    ap.add_argument("--weights", type=Path, default=None, help="Path to weights file or directory (default: auto-detect)")
    ap.add_argument("--opponent", choices=["heuristic", "random", "pool"], default="heuristic", help="Opponent type")
    ap.add_argument("--matches", type=int, default=50, help="Number of games")
    ap.add_argument("--depth", type=int, default=4, help="Minimax depth for both agents (except random)")
    ap.add_argument("--random-openings", type=int, default=4, help="Random opening plies per game")
    ap.add_argument("--no-swap", action="store_true", help="Do not swap colors between games")
    ap.add_argument("--seed", type=int, default=None, help="Random seed")
    ap.add_argument("--progress", action="store_true", help="Print per-game progress updates")
    ap.add_argument("--opp-dir", type=Path, default=Path("weights"), help="Directory to sample opponent weights when using pool")
    ap.add_argument("--opp-order", choices=["newest", "oldest", "name"], default="newest", help="Order of opponent selection in pool mode")
    ap.add_argument("--opp-limit", type=int, default=0, help="Limit the number of opponents from the pool (0 = all)")
    ap.add_argument("--opp-skip-newest", type=int, default=0, help="Skip the N newest pool entries (e.g., 1 to face previous final)")
    ap.add_argument("--epsilon-move", type=float, default=0.0, help="Per-move epsilon to add randomness to move selection")
    ap.add_argument("--matches-per-opp", type=int, default=0, help="Override to play fixed matches per opponent (pool mode)")
    ap.add_argument("--workers", type=int, default=1, help="Parallel workers for evaluation (games in parallel)")
    ap.add_argument("--parallel", choices=["process", "thread"], default="process", help="Parallel backend for games")
    args = ap.parse_args()

    params = load_params_from_candidates(args.weights)
    cfg = EvalConfig(
        matches=args.matches,
        depth=args.depth,
        random_openings=args.random_openings,
        swap_colors=not args.no_swap,
        seed=args.seed,
    )
    if args.progress:
        print(
            f"[eval] Start: requested_matches={cfg.matches}, depth={cfg.depth}, random_openings={cfg.random_openings}, opponent={args.opponent}",
            flush=True,
        )
    res = evaluate(
        params,
        opponent_kind=args.opponent,
        cfg=cfg,
        progress=args.progress,
        opp_dir=args.opp_dir,
        opp_order=args.opp_order,
        opp_limit=args.opp_limit,
        matches_per_opp=args.matches_per_opp,
        workers=args.workers,
        opp_skip_newest=args.opp_skip_newest,
        epsilon_move=args.epsilon_move,
        parallel=args.parallel,
    )
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
