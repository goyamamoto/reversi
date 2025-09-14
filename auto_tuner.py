from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
import datetime as dt
from pathlib import Path
from typing import Optional

from eval_agents import evaluate, load_params_from_candidates, corner_audit, EvalConfig
from eval_params import Params
import train_q


def list_final_versions(save_dir: Path) -> list[Path]:
    """Return versioned final weights (timestamped files ending with 'ep*.json'), newest first."""
    if not save_dir.exists():
        return []
    files = [p for p in save_dir.glob("weights-*ep*.json") if p.is_file()]
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return files


def previous_final_versions(save_dir: Path, k: int = 2) -> list[Path]:
    files = list_final_versions(save_dir)
    # Skip the most recent (assumed current), return previous k
    prev = files[1 : 1 + k]
    return prev


def previous_final_version(save_dir: Path) -> Optional[Path]:
    files = list_final_versions(save_dir)
    if len(files) >= 2:
        return files[1]
    return None

@dataclass
class TuneConfig:
    cycles: int = 5
    episodes_per_cycle: int = 1000
    depth: int = 4
    matches: int = 40
    random_openings_eval: int = 6
    min_improve: float = 0.02  # 2% absolute win-rate improvement
    reduce_alpha_on_fail: float = 0.5
    bump_epsilon_on_fail: float = 1.2
    bump_l2_on_fail: float = 2.0
    promote_best: bool = True  # update latest.json to best after each cycle
    until_corner: bool = False
    corner_threshold: float = 0.7
    corner_min_opportunities: int = 20
    increase_random_openings_on_low_opps: int = 2
    max_random_openings_train: int = 12
    episodes_mult_on_fail: float = 1.0
    # Safety bounds
    eps_min: float = 0.02
    eps_max: float = 0.3
    alpha_min: float = 1e-5
    l2_max: float = 1e-3
    # Corner handling
    continue_after_corner: bool = False


def run_cycle(
    save_dir: Path,
    out_path: Path,
    init_weights: Optional[Path],
    episodes: int,
    alpha: float,
    alpha_decay: float,
    gamma: float,
    epsilon: float,
    eps_decay: float,
    random_openings_train: int,
    l2: float,
    clip_params: Optional[float],
    normalize_features: bool,
    reward: str,
    depth_eval: int,
    matches: int,
    random_openings_eval: int,
    eval_workers: int = 1,
    progress_eval: bool = False,
    # learning with minimax/fit-to-search
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
    # Train for one cycle
    if progress_eval:
        print(
            f"[auto-tune] Training: episodes={episodes} alpha={alpha:.6f} eps={epsilon:.4f} l2={l2:.2e}",
            flush=True,
        )
    train_q.train(
        episodes=episodes,
        alpha=alpha,
        gamma=gamma,
        epsilon=epsilon,
        eps_decay=eps_decay,
        seed=None,
        save_path=out_path,
        alpha_decay=alpha_decay,
        normalize_features_flag=normalize_features,
        reward_type=reward,
        clip_params=clip_params,
        random_openings=random_openings_train,
        checkpoint=max(episodes // 2, 1),
        init_weights=init_weights,
        pos_scale=4.0,
        mob_scale=20.0,
        stab_scale=28.0,
        l2=l2,
        save_dir=save_dir,
        auto_version=True,
        resume_latest=init_weights is None,
        tag=None,
        update_root_latest=True,
        search_depth=search_depth,
        search_policy=search_policy,
        search_workers=search_workers,
        fit_to_search=fit_to_search,
        fit_cycles=fit_cycles,
        max_samples=max_samples,
        ridge_to_current=ridge_to_current,
        fit_play_policy=fit_play_policy,
        fit_epsilon=fit_epsilon,
    )
    # Evaluate latest vs heuristic
    params = load_params_from_candidates(save_dir)
    if progress_eval:
        print(
            f"[auto-tune] Evaluating: matches={matches} depth={depth_eval} rand_openings={random_openings_eval}",
            flush=True,
        )
    cfg = EvalConfig(matches=matches, depth=depth_eval, random_openings=random_openings_eval, swap_colors=True)
    res = evaluate(params, opponent_kind="pool", cfg=cfg, progress=progress_eval, workers=eval_workers)
    return res


def auto_tune(cfg: TuneConfig, save_dir: Path, out_path: Path, init_weights: Optional[Path], alpha: float, alpha_decay: float, gamma: float, epsilon: float, eps_decay: float, random_openings_train: int, l2: float, clip_params: Optional[float], normalize_features: bool, reward: str, progress: bool = False, eval_workers: int = 1):
    # Baseline eval before tuning
    best_params = load_params_from_candidates(init_weights if init_weights else save_dir)
    if progress:
        print(
            f"[auto-tune] Baseline evaluating: matches={cfg.matches} depth={cfg.depth} rand_openings={cfg.random_openings_eval} workers={eval_workers}",
            flush=True,
        )
    prev_final = previous_final_version(save_dir)
    best_res = (
        evaluate(
            best_params,
            opponent_kind="pool",
            cfg=EvalConfig(matches=cfg.matches, depth=cfg.depth, random_openings=cfg.random_openings_eval, swap_colors=True),
            progress=progress,
            workers=eval_workers,
            opp_paths=[prev_final] if prev_final else None,
        )
        if best_params
        else {"win_rate": 0.0, "games": 0, "wins": 0, "losses": 0, "draws": 0}
    )
    best_path: Optional[Path] = None
    if init_weights and init_weights.exists():
        best_path = init_weights
    else:
        # try to resolve current latest
        latest = save_dir / "latest.json"
        best_path = latest if latest.exists() else None

    last_corner_rate = 0.0
    i = 0
    while True:
        i += 1
        if cfg.cycles > 0 and i > cfg.cycles:
            break
        print(
            f"\n[auto-tune] Cycle {i}/{cfg.cycles}: alpha={alpha:.6f}, epsilon={epsilon:.4f}, l2={l2:.2e}",
            flush=True,
        )
        res = run_cycle(
            save_dir=save_dir,
            out_path=out_path,
            init_weights=best_path,
            episodes=cfg.episodes_per_cycle,
            alpha=alpha,
            alpha_decay=alpha_decay,
            gamma=gamma,
            epsilon=epsilon,
            eps_decay=eps_decay,
            random_openings_train=random_openings_train,
            l2=l2,
            clip_params=clip_params,
            normalize_features=normalize_features,
            reward=reward,
            depth_eval=cfg.depth,
            matches=cfg.matches,
            random_openings_eval=cfg.random_openings_eval,
            eval_workers=eval_workers,
            progress_eval=progress,
            search_depth=args.search_depth,
            search_policy=args.search_policy,
            search_workers=args.search_workers,
            fit_to_search=args.fit_to_search,
            fit_cycles=args.fit_cycles,
            max_samples=args.max_samples,
            ridge_to_current=args.ridge_to_current,
            fit_play_policy=args.fit_play_policy,
            fit_epsilon=args.fit_epsilon,
        )
        new_wr = res["win_rate"]
        print(f"[auto-tune] Result (pool): {res}")
        # Load latest params once for anchor and corner audit
        params_after = load_params_from_candidates(save_dir)
        # Evaluate vs the immediately previous final version to monitor learning effect
        prev_final = previous_final_version(save_dir)
        if prev_final:
            prev_eval = evaluate(
                params_after,
                opponent_kind="pool",
                cfg=EvalConfig(
                    matches=min(cfg.matches, 40), depth=cfg.depth, random_openings=cfg.random_openings_eval, swap_colors=True
                ),
                progress=False,
                workers=eval_workers,
                opp_paths=[prev_final],
            )
            print(f"[auto-tune] Vs previous final: {prev_final.name} -> {prev_eval}")
        # Also report vs heuristic as a stable anchor
        anchor = evaluate(
            params_after,
            opponent_kind="heuristic",
            cfg=EvalConfig(
                matches=cfg.matches, depth=cfg.depth, random_openings=cfg.random_openings_eval, swap_colors=True
            ),
            progress=False,
        )
        print(f"[auto-tune] Anchor vs heuristic: {anchor}")

        # Corner audit
        if progress:
            print(
                f"[auto-tune] Corner audit: matches={cfg.matches} depth={cfg.depth} rand_openings={cfg.random_openings_eval}",
                flush=True,
            )
        corner = corner_audit(
            params_after,
            opponent_kind="pool",
            cfg=EvalConfig(matches=cfg.matches, depth=cfg.depth, random_openings=cfg.random_openings_eval, swap_colors=True),
            opp_dir=save_dir,
            opp_order="newest",
            opp_limit=5,
            matches_per_opp=0,
            progress=progress,
            workers=eval_workers,
        )
        print(f"[auto-tune] Corner audit: {corner}")

        if cfg.until_corner and corner["opportunities"] >= cfg.corner_min_opportunities and corner["rate"] >= cfg.corner_threshold:
            # Save milestone snapshot
            try:
                params_after = load_params_from_candidates(save_dir)
                if params_after is not None:
                    ts = dt.datetime.now(dt.UTC).strftime("%Y%m%d-%H%M%S")
                    milestone_dir = save_dir / "milestones"
                    milestone_dir.mkdir(parents=True, exist_ok=True)
                    meta = {
                        "schema": 1,
                        "created_at": dt.datetime.now(dt.UTC).isoformat().replace("+00:00", "Z"),
                        "type": "corner_milestone",
                        "cycle": i,
                        "corner_rate": corner["rate"],
                        "corner_opportunities": corner["opportunities"],
                        "matches": cfg.matches,
                        "depth": cfg.depth,
                        "random_openings_eval": cfg.random_openings_eval,
                    }
                    milestone_path = milestone_dir / f"corner-{ts}-rate{int(corner['rate']*100)}-opps{corner['opportunities']}.json"
                    milestone_path.write_text(train_q._dump_params_json(params_after, meta))
                    print(
                        f"[auto-tune] First milestone reached: corner preference >= {cfg.corner_threshold:.0%}"
                        f" (rate={corner['rate']:.1%}, opps={corner['opportunities']}) at cycle {i}.\n"
                        f"[auto-tune] Saved milestone to {milestone_path}",
                        flush=True,
                    )
            except Exception as e:
                print(f"[auto-tune] Failed to save corner milestone: {e}", flush=True)
            if cfg.continue_after_corner:
                print("[auto-tune] Corner milestone reached. Continuing as requested.")
            else:
                print("[auto-tune] Corner preference threshold reached. Stopping.")
                break

        if new_wr >= best_res.get("win_rate", 0.0) + cfg.min_improve:
            print(f"[auto-tune] Improvement found: {best_res.get('win_rate', 0.0):.3f} -> {new_wr:.3f}")
            best_res = res
            # promote latest as best
            best_path = save_dir / "latest.json"
        else:
            print("[auto-tune] No sufficient improvement. Adjusting hyperparameters.")
            alpha = max(cfg.alpha_min, alpha * cfg.reduce_alpha_on_fail)
            epsilon = min(cfg.eps_max, max(cfg.eps_min, epsilon * cfg.bump_epsilon_on_fail))
            l2 = min(cfg.l2_max, l2 * cfg.bump_l2_on_fail)
            # If not enough corner opportunities, increase random openings during training
            if corner["opportunities"] < cfg.corner_min_opportunities:
                random_openings_train = min(cfg.max_random_openings_train, random_openings_train + cfg.increase_random_openings_on_low_opps)
                print(f"[auto-tune] Increased random_openings_train to {random_openings_train}")
            # Optionally increase episodes per cycle if corner rate stalls
            if cfg.episodes_mult_on_fail and cfg.episodes_mult_on_fail > 1.0 and corner["rate"] <= last_corner_rate:
                cfg.episodes_per_cycle = int(cfg.episodes_per_cycle * cfg.episodes_mult_on_fail)
                print(f"[auto-tune] Increased episodes_per_cycle to {cfg.episodes_per_cycle}")
            # revert latest to previous best if available
            if cfg.promote_best and best_path and best_path.exists():
                try:
                    shutil.copy2(best_path, save_dir / "latest.json")
                    shutil.copy2(best_path, Path("weights.json"))
                    print(f"[auto-tune] Reverted latest to {best_path}")
                except Exception as e:
                    print(f"[auto-tune] Failed to revert latest: {e}")
        last_corner_rate = corner["rate"]

    print(f"\n[auto-tune] Finished. Best win_rate={best_res.get('win_rate', 0.0):.3f}")


def main():
    ap = argparse.ArgumentParser(description="Auto tuner: train -> evaluate -> (corner audit) -> adjust -> repeat")
    ap.add_argument("--cycles", type=int, default=5, help="Number of train/eval cycles (0 = loop forever)")
    ap.add_argument("--episodes", type=int, default=1000, help="Episodes per cycle")
    ap.add_argument("--save-dir", type=Path, default=Path("weights"), help="Directory to save versioned weights")
    ap.add_argument("--out", type=Path, default=Path("weights.json"), help="Root output path to update")
    ap.add_argument("--init-weights", type=Path, default=None, help="Start from given weights (otherwise latest)")

    # Training hyperparameters (initial)
    ap.add_argument("--alpha", type=float, default=0.01, help="Initial learning rate")
    ap.add_argument("--alpha-decay", type=float, default=0.999, help="Learning rate decay per episode")
    ap.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    ap.add_argument("--epsilon", type=float, default=0.2, help="Initial exploration rate")
    ap.add_argument("--eps-decay", type=float, default=0.995, help="Exploration decay per episode")
    ap.add_argument("--random-openings-train", type=int, default=6, help="Random opening plies during training")
    ap.add_argument("--l2", type=float, default=1e-4, help="L2 regularization (weight decay)")
    ap.add_argument("--clip-params", type=float, default=1.0, help="Parameter clip bound")
    ap.add_argument("--normalize-features", action="store_true", help="Normalize features")
    ap.add_argument("--reward", choices=["winloss", "margin"], default="winloss", help="Terminal reward type")

    # Evaluation
    ap.add_argument("--depth", type=int, default=4, help="Minimax depth for evaluation")
    ap.add_argument("--matches", type=int, default=40, help="Games per evaluation")
    ap.add_argument("--random-openings-eval", type=int, default=6, help="Random opening plies during evaluation")

    # Policy
    ap.add_argument("--min-improve", type=float, default=0.02, help="Min absolute win-rate improvement to accept")
    ap.add_argument("--reduce-alpha-on-fail", type=float, default=0.5, help="Factor to reduce alpha if not improved")
    ap.add_argument("--bump-epsilon-on-fail", type=float, default=1.2, help="Multiply epsilon if not improved")
    ap.add_argument("--bump-l2-on-fail", type=float, default=2.0, help="Multiply L2 if not improved")
    ap.add_argument("--no-promote-best", action="store_true", help="Do not revert latest to best on failure")
    # Corner preference options
    ap.add_argument("--until-corner", action="store_true", help="Track corner preference and optionally stop at threshold")
    ap.add_argument("--continue-after-corner", action="store_true", help="Record corner milestones but keep training (used with --until-corner)")
    ap.add_argument("--corner-threshold", type=float, default=0.7, help="Target corner-take rate when a corner is available")
    ap.add_argument("--corner-min-opportunities", type=int, default=20, help="Minimum corner opportunities required to judge")
    ap.add_argument("--increase-random-openings-on-low-opps", type=int, default=2, help="Increase training random openings when corner opportunities are low")
    ap.add_argument("--max-random-openings-train", type=int, default=12, help="Maximum random openings during training")
    ap.add_argument("--episodes-mult-on-fail", type=float, default=1.0, help="Multiply episodes per cycle when corner rate stalls (>1 to enable)")
    ap.add_argument("--eval-workers", type=int, default=1, help="Parallel workers for evaluation and corner audit")
    # Learning with search (train_q)
    ap.add_argument("--search-depth", type=int, default=2, help="Depth for minimax-backed targets (0 disables)")
    ap.add_argument("--search-policy", action="store_true", help="Choose actions by minimax (with epsilon exploration)")
    ap.add_argument("--search-workers", type=int, default=1, help="Parallel workers for minimax during training")
    ap.add_argument("--fit-to-search", action="store_true", help="Fit linear evaluator to minimax(search-depth) targets (supervised)")
    ap.add_argument("--fit-cycles", type=int, default=1, help="Repeat fit-to-search this many times per cycle")
    ap.add_argument("--max-samples", type=int, default=1000, help="Max samples (states) per fit cycle")
    ap.add_argument("--ridge-to-current", action="store_true", help="Ridge toward current weights instead of zero in fit")
    ap.add_argument("--fit-play-policy", choices=["minimax", "epsilon", "random"], default="minimax", help="Sampling policy for fit-to-search")
    ap.add_argument("--fit-epsilon", type=float, default=0.1, help="Exploration epsilon for sampling in fit-to-search")
    ap.add_argument("--progress", action="store_true", help="Show progress during train/eval/audit phases")

    args = ap.parse_args()

    if args.progress:
        print(
            f"[auto-tune] Start: cycles={args.cycles} episodes/cycle={args.episodes} depth={args.depth} matches={args.matches}",
            flush=True,
        )

    cfg = TuneConfig(
        cycles=args.cycles,
        episodes_per_cycle=args.episodes,
        depth=args.depth,
        matches=args.matches,
        random_openings_eval=args.random_openings_eval,
        min_improve=args.min_improve,
        reduce_alpha_on_fail=args.reduce_alpha_on_fail,
        bump_epsilon_on_fail=args.bump_epsilon_on_fail,
        bump_l2_on_fail=args.bump_l2_on_fail,
        promote_best=not args.no_promote_best,
        until_corner=args.until_corner,
        corner_threshold=args.corner_threshold,
        corner_min_opportunities=args.corner_min_opportunities,
        increase_random_openings_on_low_opps=args.increase_random_openings_on_low_opps,
        max_random_openings_train=args.max_random_openings_train,
        episodes_mult_on_fail=args.episodes_mult_on_fail,
        continue_after_corner=args.continue_after_corner,
    )

    auto_tune(
        cfg=cfg,
        save_dir=args.save_dir,
        out_path=args.out,
        init_weights=args.init_weights,
        alpha=args.alpha,
        alpha_decay=args.alpha_decay,
        gamma=args.gamma,
        epsilon=args.epsilon,
        eps_decay=args.eps_decay,
        random_openings_train=args.random_openings_train,
        l2=args.l2,
        clip_params=args.clip_params,
        normalize_features=args.normalize_features,
        reward=args.reward,
        progress=args.progress,
        eval_workers=args.eval_workers,
    )


if __name__ == "__main__":
    main()
