import argparse
import json
import random
from pathlib import Path
import datetime as dt
from typing import Optional

from eval_params import Params, evaluate_with_params, compute_pos_features, count_edge_stable, opponent
from main import Board, BLACK, WHITE


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
        params = Params.zeros()

    for ep in range(1, episodes + 1):
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
            # TD(0) on state values with self-play (black-fixed)
            pos_f, mob_f, stab_f = features_state_diff_black(board)
            pos_f_n, mob_f_n, stab_f_n = maybe_normalize_features(
                pos_f, mob_f, stab_f, normalize_features_flag, pos_scale, mob_scale, stab_scale
            )
            v_pred = value_from_features(pos_f_n, mob_f_n, stab_f_n, params)

            last_player = cur_player
            # choose action and step
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

        if checkpoint and ep % checkpoint == 0:
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

        if ep % max(1, episodes // 10) == 0 or ep == 1:
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
    )


if __name__ == "__main__":
    main()
