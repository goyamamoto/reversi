import argparse
import json
import random
from pathlib import Path
import datetime as dt
from typing import Optional, List
import os
import concurrent.futures as cf

from eval_params import (
    Params,
    evaluate_with_params,
    compute_pos_features,
    count_edge_stable,
    opponent,
    evaluate_full_black_advantage,
    sym_index,
)
from main import Board, BLACK, WHITE, MinimaxAI


def epsilon_greedy_action(
    board: Board,
    player: int,
    params: Params,
    epsilon: float,
    normalize_features_flag: bool,
    pos_scale: float,
    mob_scale: float,
    stab_scale: float,
):
    moves = board.valid_moves(player)
    if not moves:
        return None
    if random.random() < epsilon:
        return random.choice(moves)
    # pick move that maximizes Black advantage after the move
    best = None
    best_val = float('-inf')
    for m in moves:
        nb = board.copy()
        nb.apply_move(m, player)
        # Evaluate next state as Black advantage (turn-agnostic)
        pos_f, mob_f, stab_f = features_state_diff_black(nb)
        pos_f, mob_f, stab_f = maybe_normalize_features(
            pos_f, mob_f, stab_f, normalize_features_flag, pos_scale, mob_scale, stab_scale
        )
        val = value_from_features(pos_f, mob_f, stab_f, params)
        if player == WHITE:
            val = -val
        if val > best_val:
            best_val = val
            best = m
    return best


def features_state(board: Board, player: int):
    # Legacy per-player features (kept for compatibility)
    pos = compute_pos_features(board.grid, player)
    mob = len(board.valid_moves(player))
    stab = count_edge_stable(board.grid, player)
    return pos, mob, stab


def features_state_diff_black(board: Board):
    """Zero-sum, black-fixed features: (Black-White) for all components.

    - Positional quadrant features: compute_pos_features with player=BLACK (already B-W)
    - Mobility: len(moves(BLACK)) - len(moves(WHITE))
    - Edge-stable: count_edge_stable(BLACK) - count_edge_stable(WHITE)
    """
    pos = compute_pos_features(board.grid, BLACK)
    mob = len(board.valid_moves(BLACK)) - len(board.valid_moves(WHITE))
    stab = count_edge_stable(board.grid, BLACK) - count_edge_stable(board.grid, WHITE)
    return pos, mob, stab


def value_from_features(pos, mob, stab, params: Params):
    v = sum(p * w for p, w in zip(pos, params.pos))
    v += params.mobility_w * mob
    v += params.stable_w * stab
    return v


def maybe_normalize_features(pos, mob, stab, normalize: bool, pos_scale: float, mob_scale: float, stab_scale: float):
    if not normalize:
        return pos, mob, stab
    # pos feature per index is in [-4,4] (4 symmetric cells), scale to roughly [-1,1]
    pos_n = [p / 4.0 for p in pos]
    mob_n = mob / mob_scale  # default 20.0
    stab_n = stab / stab_scale  # default 28.0 (perimeter cells)
    return pos_n, mob_n, stab_n


# ------------ Minimax for training (black-advantage) ------------
def _leaf_value_black_adv(board: Board, params: Params, normalize: bool, pos_scale: float, mob_scale: float, stab_scale: float) -> float:
    # Terminal handling: encode win/loss as full-board equivalents
    try:
        is_terminal = (len(board.valid_moves(BLACK)) == 0) and (len(board.valid_moves(WHITE)) == 0)
    except Exception:
        is_terminal = False
    if is_terminal:
        b = sum(cell == BLACK for row in board.grid for cell in row)
        w = sum(cell == WHITE for row in board.grid for cell in row)
        if b > w:
            # full-Black board value
            if normalize:
                pos, mob, stab = [1.0] * 16, 0.0, 1.0
            else:
                pos, mob, stab = [4.0] * 16, 0.0, 28.0
            return value_from_features(pos, mob, stab, params)
        elif w > b:
            if normalize:
                pos, mob, stab = [-1.0] * 16, -0.0, -1.0
            else:
                pos, mob, stab = [-4.0] * 16, -0.0, -28.0
            return value_from_features(pos, mob, stab, params)
        else:
            return 0.0
    # Non-terminal: standard evaluation of current board
    pos, mob, stab = features_state_diff_black(board)
    pos, mob, stab = maybe_normalize_features(pos, mob, stab, normalize, pos_scale, mob_scale, stab_scale)
    return value_from_features(pos, mob, stab, params)


def minimax_value_black_adv(
    board: Board,
    params: Params,
    depth: int,
    to_move: int,
    normalize: bool,
    pos_scale: float,
    mob_scale: float,
    stab_scale: float,
    alpha: float = float("-inf"),
    beta: float = float("inf"),
    workers: int = 1,
) -> float:
    # Terminal or depth limit
    my_moves = board.valid_moves(to_move)
    opp = opponent(to_move)
    opp_moves = board.valid_moves(opp)
    if depth <= 0 or board.full() or (not my_moves and not opp_moves):
        return _leaf_value_black_adv(board, params, normalize, pos_scale, mob_scale, stab_scale)

    # Handle pass
    if not my_moves:
        return minimax_value_black_adv(board, params, depth - 1, opp, normalize, pos_scale, mob_scale, stab_scale, alpha, beta)

    # Parallelize one ply if workers>1 and branching factor>1
    if workers and workers > 1 and len(my_moves) > 1 and depth > 1:
        specs = []
        for m in my_moves:
            nb = board.copy()
            nb.apply_move(m, to_move)
            specs.append((nb, params, depth - 1, opp, normalize, pos_scale, mob_scale, stab_scale, alpha, beta, 1))
        with cf.ProcessPoolExecutor(max_workers=workers) as ex:
            vals = list(
                ex.map(
                    _minimax_value_black_adv_standalone,
                    [s[0] for s in specs],
                    [s[1] for s in specs],
                    [s[2] for s in specs],
                    [s[3] for s in specs],
                    [s[4] for s in specs],
                    [s[5] for s in specs],
                    [s[6] for s in specs],
                    [s[7] for s in specs],
                    [s[8] for s in specs],
                    [s[9] for s in specs],
                    [s[10] for s in specs],
                )
            )
        if to_move == BLACK:
            return max(vals)
        else:
            return min(vals)
    # Fallback serial recursion
    if to_move == BLACK:
        best = float("-inf")
        for m in my_moves:
            nb = board.copy()
            nb.apply_move(m, to_move)
            val = minimax_value_black_adv(nb, params, depth - 1, opp, normalize, pos_scale, mob_scale, stab_scale, alpha, beta, workers)
            if val > best:
                best = val
            if best > alpha:
                alpha = best
            if alpha >= beta:
                break
        return best
    else:
        best = float("inf")
        for m in my_moves:
            nb = board.copy()
            nb.apply_move(m, to_move)
            val = minimax_value_black_adv(nb, params, depth - 1, opp, normalize, pos_scale, mob_scale, stab_scale, alpha, beta, workers)
            if val < best:
                best = val
            if best < beta:
                beta = best
            if alpha >= beta:
                break
        return best


def _minimax_value_black_adv_standalone(
    board: Board,
    params: Params,
    depth: int,
    to_move: int,
    normalize: bool,
    pos_scale: float,
    mob_scale: float,
    stab_scale: float,
    alpha: float,
    beta: float,
    workers: int,
) -> float:
    return minimax_value_black_adv(
        board, params, depth, to_move, normalize, pos_scale, mob_scale, stab_scale, alpha, beta, workers
    )


# ------------ Supervised fit to minimax targets (batch regression) ------------

def _feature_vector_black(board: Board, to_move: int, normalize: bool, pos_scale: float, mob_scale: float, stab_scale: float) -> List[float]:
    """Return feature vector that encodes (state, turn) for black-advantage model.

    We use black-minus-white features and multiply by +1 when BLACK to move,
    -1 when WHITE to move, so that a single linear function w·x can represent
    V(s, to_move) consistently for both turns.
    """
    pos, mob, stab = features_state_diff_black(board)
    pos, mob, stab = maybe_normalize_features(pos, mob, stab, normalize, pos_scale, mob_scale, stab_scale)
    sign = 1.0 if to_move == BLACK else -1.0
    # Flatten into 18-dim vector: 16 pos + mob + stab, with turn sign
    vec = [sign * p for p in pos]
    vec.append(sign * mob)
    vec.append(sign * stab)
    return vec


def _mat_vec_mul_T(X: List[List[float]], y: List[float]) -> List[float]:
    # return X^T y
    n = len(X[0])
    out = [0.0] * n
    for row, vy in zip(X, y):
        for j in range(n):
            out[j] += row[j] * vy
    return out


def _mat_T_mat(X: List[List[float]]) -> List[List[float]]:
    # return X^T X (n x n)
    n = len(X[0])
    A = [[0.0 for _ in range(n)] for _ in range(n)]
    for row in X:
        for i in range(n):
            xi = row[i]
            if xi == 0.0:
                continue
            for j in range(n):
                A[i][j] += xi * row[j]
    return A


def _cholesky_solve(A: List[List[float]], b: List[float]) -> List[float]:
    # Solve Ax=b for SPD A via Cholesky (naive, small n)
    n = len(A)
    # Copy
    L = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1):
            s = A[i][j]
            for k in range(j):
                s -= L[i][k] * L[j][k]
            if i == j:
                L[i][j] = (s if s > 0 else 1e-12) ** 0.5
            else:
                L[i][j] = s / (L[j][j] if L[j][j] != 0 else 1e-12)
    # Solve Ly = b
    y = [0.0] * n
    for i in range(n):
        s = b[i]
        for k in range(i):
            s -= L[i][k] * y[k]
        y[i] = s / (L[i][i] if L[i][i] != 0 else 1e-12)
    # Solve L^T x = y
    x = [0.0] * n
    for i in reversed(range(n)):
        s = y[i]
        for k in range(i + 1, n):
            s -= L[k][i] * x[k]
        x[i] = s / (L[i][i] if L[i][i] != 0 else 1e-12)
    return x


def fit_to_minimax(
    episodes: int,
    params: Params,
    depth: int,
    normalize: bool,
    pos_scale: float,
    mob_scale: float,
    stab_scale: float,
    random_openings: int,
    l2: float,
    ridge_to_current: bool,
    workers: int,
    max_samples: int,
    play_policy: str = "minimax",
    sample_epsilon: float = 0.1,
) -> Params:
    print(
        f"[fit] start: episodes={episodes} depth={depth} workers={workers} max_samples={max_samples}",
        flush=True,
    )
    # Collect dataset
    X: List[List[float]] = []
    y: List[float] = []
    for ep in range(episodes):
        board = Board()
        cur = BLACK
        terminal = False
        ro = random_openings
        while ro > 0 and not terminal:
            moves = board.valid_moves(cur)
            if not moves:
                cur = opponent(cur)
                if not board.valid_moves(cur):
                    terminal = True
                ro -= 1
                continue
            m = random.choice(moves)
            board.apply_move(m, cur)
            cur = opponent(cur)
            ro -= 1

        while not terminal and len(X) < max_samples:
            # Sample this state
            X.append(_feature_vector_black(board, cur, normalize, pos_scale, mob_scale, stab_scale))
            # Minimax target from current state (black-advantage scalar)
            t = minimax_value_black_adv(
                board,
                params,
                depth,
                cur,
                normalize,
                pos_scale,
                mob_scale,
                stab_scale,
                workers=workers,
            )
            # Align teacher with feature sign convention: model is V(s,turn)=sign(turn)*V_black_adv(s)
            t_signed = t if cur == BLACK else -t
            y.append(t_signed)
            if len(X) % 50 == 0:
                print(f"[fit] samples: {len(X)}/{max_samples}", flush=True)
            # Advance 1 ply using pseudo-play policy
            moves = board.valid_moves(cur)
            if not moves:
                cur = opponent(cur)
                if not board.valid_moves(cur):
                    terminal = True
                continue
            m = None
            r = random.random()
            if play_policy == "minimax" and depth > 0 and (r >= sample_epsilon):
                m = choose_minimax_move(
                    board,
                    cur,
                    params,
                    depth,
                    normalize,
                    pos_scale,
                    mob_scale,
                    stab_scale,
                    workers=workers,
                )
            elif play_policy == "epsilon":
                m = epsilon_greedy_action(
                    board,
                    cur,
                    params,
                    sample_epsilon,
                    normalize,
                    pos_scale,
                    mob_scale,
                    stab_scale,
                )
            # fallback random if policy returned None or selected by epsilon
            if m is None:
                m = random.choice(moves)
            board.apply_move(m, cur)
            cur = opponent(cur)

        if len(X) >= max_samples:
            break

    if not X:
        return params

    n = len(X[0])  # 18 dims
    A = _mat_T_mat(X)
    bvec = _mat_vec_mul_T(X, y)

    # Ridge regularization
    for i in range(n):
        A[i][i] += l2
    if ridge_to_current:
        # Bias toward current weights
        w0 = params.pos + [params.mobility_w, params.stable_w]
        for i in range(n):
            bvec[i] += l2 * w0[i]

    w = _cholesky_solve(A, bvec)
    new_params = Params(pos=w[:16], mobility_w=w[16], stable_w=w[17])
    return new_params


def choose_minimax_move(
    board: Board,
    player: int,
    params: Params,
    depth: int,
    normalize: bool,
    pos_scale: float,
    mob_scale: float,
    stab_scale: float,
    workers: int = 1,
):
    moves = board.valid_moves(player)
    if not moves:
        return None
    opp = opponent(player)
    best_move = None
    best_val = float("-inf")
    if workers and workers > 1 and len(moves) > 1 and depth > 1:
        specs = []
        for m in moves:
            nb = board.copy()
            nb.apply_move(m, player)
            specs.append((nb, params, depth - 1, opp, normalize, pos_scale, mob_scale, stab_scale, float("-inf"), float("inf"), workers))
        with cf.ProcessPoolExecutor(max_workers=workers) as ex:
            vals = list(
                ex.map(
                    _minimax_value_black_adv_standalone,
                    [s[0] for s in specs],
                    [s[1] for s in specs],
                    [s[2] for s in specs],
                    [s[3] for s in specs],
                    [s[4] for s in specs],
                    [s[5] for s in specs],
                    [s[6] for s in specs],
                    [s[7] for s in specs],
                    [s[8] for s in specs],
                    [s[9] for s in specs],
                    [s[10] for s in specs],
                )
            )
        for m, v in zip(moves, vals):
            if player == WHITE:
                v = -v
            if v > best_val:
                best_val = v
                best_move = m
    else:
        for m in moves:
            nb = board.copy()
            nb.apply_move(m, player)
            # value from child's position; next to move is opponent
            v = minimax_value_black_adv(nb, params, depth - 1, opp, normalize, pos_scale, mob_scale, stab_scale, workers=workers)
            # Convert to current player's perspective from black-advantage scalar
            if player == WHITE:
                v = -v
            if v > best_val:
                best_val = v
                best_move = m
    return best_move


def _find_latest_weight(save_dir: Path) -> Optional[Path]:
    if not save_dir.exists() or not save_dir.is_dir():
        return None
    cands = sorted([p for p in save_dir.glob("*.json") if p.is_file()], key=lambda p: p.stat().st_mtime, reverse=True)
    return cands[0] if cands else None


def _dump_params_json(params: Params, meta: dict) -> str:
    d = params.to_dict()
    d["__meta__"] = meta
    return json.dumps(d, indent=2)


def train(
    episodes: int,
    alpha: float,
    gamma: float,
    epsilon: float,
    eps_decay: float,
    seed: Optional[int],
    save_path: Path,
    alpha_decay: float,
    normalize_features_flag: bool,
    reward_type: str,
    clip_params: Optional[float],
    random_openings: int,
    checkpoint: int,
    init_weights: Optional[Path],
    pos_scale: float,
    mob_scale: float,
    stab_scale: float,
    l2: float,
    save_dir: Optional[Path],
    auto_version: bool,
    resume_latest: bool,
    tag: Optional[str],
    update_root_latest: bool,
    search_depth: int = 0,
    search_policy: bool = False,
    search_workers: int = 1,
    fit_to_search: bool = False,
    fit_cycles: int = 1,
    max_samples: int = 2000,
    ridge_to_current: bool = True,
    fit_play_policy: str = "minimax",
    fit_epsilon: float = 0.1,
):
    if seed is not None:
        random.seed(seed)

    if resume_latest and not init_weights and save_dir is not None:
        latest = _find_latest_weight(save_dir)
        if latest is not None:
            init_weights = latest
            print(f"Resuming from latest weights: {init_weights}", flush=True)

    if init_weights and init_weights.exists():
        try:
            params = Params.from_dict(json.loads(init_weights.read_text()))
            print(f"Initialized params from {init_weights}", flush=True)
        except Exception as e:
            print(f"Failed to load {init_weights}: {e}. Starting from zeros.", flush=True)
            params = Params.zeros()
    else:
        # Heuristic default initialization (aggregate POS_WEIGHTS by 4x4 symmetric groups)
        try:
            # Aggregate 8x8 positional weights into 4x4 symmetric groups by average
            group_sum = [0.0] * 16
            group_cnt = [0] * 16
            for y in range(8):
                for x in range(8):
                    idx = sym_index(x, y)
                    group_sum[idx] += MinimaxAI.POS_WEIGHTS[y][x]
                    group_cnt[idx] += 1
            pos = [group_sum[i] / max(1, group_cnt[i]) for i in range(16)]
            # Modest initial weights for mobility and stable discs (normalized scale)
            mobility_w = 0.5
            stable_w = 1.0
            params = Params(pos=pos, mobility_w=mobility_w, stable_w=stable_w)
            print("Initialized heuristic params (pos/mobility/stable)", flush=True)
        except Exception:
            params = Params.zeros()
            print("Heuristic init failed; falling back to zeros", flush=True)

    if fit_to_search and search_depth and episodes > 0:
        # Supervised fit cycles
        for c in range(fit_cycles):
            params = fit_to_minimax(
                episodes=episodes,
                params=params,
                depth=search_depth,
                normalize=normalize_features_flag,
                pos_scale=pos_scale,
                mob_scale=mob_scale,
                stab_scale=stab_scale,
                random_openings=random_openings,
                l2=l2,
                ridge_to_current=ridge_to_current,
                workers=search_workers,
                max_samples=max_samples,
                play_policy=fit_play_policy,
                sample_epsilon=fit_epsilon,
            )
            print(f"[fit] cycle {c+1}/{fit_cycles} done: updated params", flush=True)
        # Skip TD loop; proceed to saving
    else:
        for ep in range(1, episodes + 1):
            board = Board()
            cur_player = BLACK  # Black starts
            terminal = False
        board = Board()
        cur_player = BLACK  # Black starts
        terminal = False

        # random opening plies to diversify
        ro = random_openings
        while ro > 0 and not terminal:
            moves = board.valid_moves(cur_player)
            if not moves:
                cur_player = opponent(cur_player)
                if not board.valid_moves(cur_player):
                    terminal = True
                ro -= 1
                continue
            m = random.choice(moves)
            board.apply_move(m, cur_player)
            cur_player = opponent(cur_player)
            ro -= 1

        while not terminal:
            # TD(0) / Search-backed update on state values (black-fixed)
            pos_f, mob_f, stab_f = features_state_diff_black(board)
            pos_f_n, mob_f_n, stab_f_n = maybe_normalize_features(
                pos_f, mob_f, stab_f, normalize_features_flag, pos_scale, mob_scale, stab_scale
            )
            v_pred = value_from_features(pos_f_n, mob_f_n, stab_f_n, params)

            last_player = cur_player
            # choose action and step
            if search_policy and search_depth and random.random() >= epsilon:
                move = choose_minimax_move(
                    board,
                    cur_player,
                    params,
                    search_depth,
                    normalize_features_flag,
                    pos_scale,
                    mob_scale,
                    stab_scale,
                    workers=search_workers,
                )
            else:
                move = epsilon_greedy_action(
                    board,
                    cur_player,
                    params,
                    epsilon,
                    normalize_features_flag,
                    pos_scale,
                    mob_scale,
                    stab_scale,
                )
            if move is not None:
                board.apply_move(move, cur_player)
                next_player = opponent(cur_player)
                # handle pass
                if not board.valid_moves(next_player):
                    next_player = cur_player
                    if not board.valid_moves(next_player):
                        terminal = True
                cur_player = next_player
            else:
                # pass
                cur_player = opponent(cur_player)
                if not board.valid_moves(cur_player):
                    terminal = True

            # compute reward and target (black-advantage)
            if terminal:
                b, w = board.count()
                if reward_type == "margin":
                    # normalized disc difference (Black advantage)
                    r = (b - w) / 64.0
                else:  # winloss
                    r = 1.0 if b > w else (-1.0 if b < w else 0.0)
                target = r
            else:
                if search_depth and search_depth > 0:
                    target = minimax_value_black_adv(
                        board,
                        params,
                        search_depth,
                        cur_player,
                        normalize_features_flag,
                        pos_scale,
                        mob_scale,
                        stab_scale,
                        workers=search_workers,
                    )
                else:
                    # Evaluate next state as Black advantage; no sign flip
                    n_pos, n_mob, n_stab = features_state_diff_black(board)
                    n_pos, n_mob, n_stab = maybe_normalize_features(
                        n_pos, n_mob, n_stab, normalize_features_flag, pos_scale, mob_scale, stab_scale
                    )
                    v_next = value_from_features(n_pos, n_mob, n_stab, params)
                    target = gamma * v_next

            # update weights with optional L2 (weight decay)
            td = (target - v_pred)
            if l2 and l2 > 0.0:
                decay = 1.0 - alpha * l2
            else:
                decay = 1.0
            for i in range(16):
                params.pos[i] = params.pos[i] * decay + alpha * td * pos_f_n[i]
            params.mobility_w = params.mobility_w * decay + alpha * td * mob_f_n
            params.stable_w = params.stable_w * decay + alpha * td * stab_f_n

            if clip_params is not None and clip_params > 0:
                lo, hi = -clip_params, clip_params
                params.pos = [max(lo, min(hi, v)) for v in params.pos]
                params.mobility_w = max(lo, min(hi, params.mobility_w))
                params.stable_w = max(lo, min(hi, params.stable_w))

        # anneal schedules
        epsilon *= eps_decay
        alpha *= alpha_decay

        if not fit_to_search and checkpoint and ep % checkpoint == 0:
            ck_meta = {
                "schema": 1,
                "created_at": dt.datetime.now(dt.UTC).isoformat().replace("+00:00", "Z"),
                "episodes_in_run": ep,
                "alpha": alpha,
                "gamma": gamma,
                "epsilon": epsilon,
                "alpha_decay": alpha_decay,
                "eps_decay": eps_decay,
                "reward": reward_type,
                "random_openings": random_openings,
                "l2": l2,
                "clip_params": clip_params,
                "normalize_features": normalize_features_flag,
                "pos_scale": pos_scale,
                "mob_scale": mob_scale,
                "stab_scale": stab_scale,
                "resumed_from": str(init_weights) if init_weights else None,
                "tag": tag,
            }
            ck_base = save_path.with_name(f"{save_path.stem}-{ep}")
            ck_dir = save_dir if save_dir else save_path.parent
            ck_dir.mkdir(parents=True, exist_ok=True)
            ck_path = ck_dir / f"{ck_base.name}.json"
            ck_path.write_text(_dump_params_json(params, ck_meta))
            print(f"Checkpoint saved to {ck_path}", flush=True)

        if not fit_to_search and (ep % max(1, episodes // 10) == 0 or ep == 1):
            b, w = board.count()
            print(
                f"Episode {ep}/{episodes} done. Final B{b}:W{w}. eps={epsilon:.3f} alpha={alpha:.5f}",
                flush=True,
            )

    # Final save(s)
    final_meta = {
        "schema": 1,
        "created_at": dt.datetime.now(dt.UTC).isoformat().replace("+00:00", "Z"),
        "episodes_in_run": episodes,
        "alpha": alpha,
        "gamma": gamma,
        "epsilon": epsilon,
        "alpha_decay": alpha_decay,
        "eps_decay": eps_decay,
        "reward": reward_type,
        "random_openings": random_openings,
        "l2": l2,
        "clip_params": clip_params,
        "normalize_features": normalize_features_flag,
        "pos_scale": pos_scale,
        "mob_scale": mob_scale,
        "stab_scale": stab_scale,
        "resumed_from": str(init_weights) if init_weights else None,
        "tag": tag,
    }

    if auto_version and save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        ts = dt.datetime.now(dt.UTC).strftime("%Y%m%d-%H%M%S")
        tag_part = f"-{tag}" if tag else ""
        version_path = save_dir / f"weights-{ts}-{episodes}ep{tag_part}.json"
        version_path.write_text(_dump_params_json(params, final_meta))
        print(f"Saved versioned weights to {version_path}", flush=True)
        latest_path = save_dir / "latest.json"
        latest_path.write_text(_dump_params_json(params, final_meta))
        print(f"Updated {latest_path}", flush=True)
        if update_root_latest:
            Path("weights.json").write_text(_dump_params_json(params, final_meta))
            print("Updated ./weights.json", flush=True)
    else:
        # legacy single-path save
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path.write_text(_dump_params_json(params, final_meta))
        print(f"Saved weights to {save_path}", flush=True)


def main():
    p = argparse.ArgumentParser(description="Train parametric evaluator via TD/Q-learning (self-play)")
    p.add_argument("--episodes", type=int, default=1000, help="Number of self-play games")
    p.add_argument("--alpha", type=float, default=0.01, help="Learning rate")
    p.add_argument("--alpha-decay", type=float, default=1.0, help="Multiplicative alpha decay per episode")
    p.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    p.add_argument("--epsilon", type=float, default=0.2, help="Epsilon for exploration")
    p.add_argument("--eps-decay", type=float, default=0.995, help="Multiplicative epsilon decay per episode")
    p.add_argument("--seed", type=int, default=None, help="Random seed")
    p.add_argument("--out", type=Path, default=Path("weights.json"), help="Output JSON for learned weights")
    p.add_argument("--normalize-features", action="store_true", help="Normalize features to comparable scales")
    p.add_argument("--reward", choices=["winloss", "margin"], default="winloss", help="Terminal reward type")
    p.add_argument("--clip-params", type=float, default=None, help="Clip absolute value of parameters to this bound")
    p.add_argument("--random-openings", type=int, default=0, help="Number of random opening plies per episode")
    p.add_argument(
        "--checkpoint", type=int, default=0, help="Save intermediate weights every N episodes (0 to disable)"
    )
    p.add_argument("--init-weights", type=Path, default=None, help="Initialize from existing weights.json")
    p.add_argument("--pos-scale", type=float, default=4.0, help="Normalizer for position features (default 4)")
    p.add_argument("--mob-scale", type=float, default=20.0, help="Normalizer for mobility feature (default 20)")
    p.add_argument("--stab-scale", type=float, default=28.0, help="Normalizer for stable feature (default 28)")
    p.add_argument("--l2", type=float, default=0.0, help="L2 regularization coefficient (weight decay)")
    p.add_argument("--save-dir", type=Path, default=Path("weights"), help="Directory to store versioned weights")
    p.add_argument("--auto-version", action="store_true", help="Save versioned file and update latest.json")
    p.add_argument("--resume-latest", action="store_true", help="Resume from latest in save-dir if init not given")
    p.add_argument("--tag", type=str, default=None, help="Tag string for versioned filenames")
    p.add_argument("--no-update-root-latest", action="store_true", help="Do not update ./weights.json when auto-version")
    p.add_argument("--search-depth", type=int, default=0, help="Depth for minimax-backed targets (0 disables)")
    p.add_argument("--search-policy", action="store_true", help="Choose actions by minimax (with epsilon exploration)")
    p.add_argument("--search-workers", type=int, default=int(os.getenv("MINIMAX_WORKERS", "1")), help="Parallel workers for minimax during training")
    p.add_argument("--fit-to-search", action="store_true", help="Fit linear evaluator to minimax(search-depth) targets (supervised)")
    p.add_argument("--fit-cycles", type=int, default=1, help="Repeat fit-to-search this many times")
    p.add_argument("--max-samples", type=int, default=2000, help="Max samples (states) per fit cycle")
    p.add_argument("--ridge-to-current", action="store_true", help="Ridge toward current weights instead of zero")
    p.add_argument("--fit-play-policy", choices=["minimax", "epsilon", "random"], default="minimax", help="Policy to generate pseudo-play samples during fit")
    p.add_argument("--fit-epsilon", type=float, default=0.1, help="Exploration epsilon when generating samples (minimax/epsilon policy)")
    args = p.parse_args()

    train(
        args.episodes,
        args.alpha,
        args.gamma,
        args.epsilon,
        args.eps_decay,
        args.seed,
        args.out,
        args.alpha_decay,
        args.normalize_features,
        args.reward,
        args.clip_params,
        args.random_openings,
        args.checkpoint,
        args.init_weights,
        args.pos_scale,
        args.mob_scale,
        args.stab_scale,
        args.l2,
        args.save_dir,
        args.auto_version,
        args.resume_latest,
        args.tag,
        not args.no_update_root_latest,
        args.search_depth,
        args.search_policy,
        args.search_workers,
        args.fit_to_search,
        args.fit_cycles,
        args.max_samples,
        args.ridge_to_current,
        args.fit_play_policy,
        args.fit_epsilon,
    )


if __name__ == "__main__":
    main()
