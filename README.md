# Reversi (Othello) with Pygame

A simple Reversi/Othello implementation using Pygame.

## Run

Prerequisites:

- Python 3.9+
- Pygame installed in your environment: `pip install pygame`

Launch the game:

```
python main.py
```

## Controls

- Left Click: place a disc on a valid cell
- R: restart a new game
- H: toggle valid-move hints
- A: toggle Minimax AI for White (Black is human)
- L: toggle learned evaluator vs heuristic (if `weights.json` exists)
- [: decrease Minimax depth, ]: increase depth (1–8)
- ESC or Q: quit

## Rules

- Standard 8x8 Reversi rules
- If current player has no valid moves, the turn automatically passes
- Game ends when neither player has a valid move or the board is full

## Notes

- The built-in AI uses Minimax with alpha-beta pruning (default depth 3).
- Depth can be adjusted at runtime with `[` and `]` keys; default is 4.
- If `weights.json` exists (trained by the script below), Minimax will use a parametric evaluator with those learned weights.

## Train Parametric Evaluator (Q-learning/TD)

You can train evaluation parameters via self-play TD/Q-learning. Parameters include:

- Positional weights for the 4x4 quadrant (mirrored over the board)
- Mobility weight: number of your legal moves
- Stable weight: number of (edge-anchored) stable discs you own

Run training:

```
python train_q.py --episodes 2000 --alpha 0.01 --gamma 0.99 --epsilon 0.2 --eps-decay 0.995 \
  --normalize-features --reward margin --alpha-decay 0.999 --random-openings 4 --checkpoint 500
```

This generates `weights.json`. Restart the game to use the learned parameters.

For stronger corner preference, try increasing search depth and emphasize stable discs during training:

```
python train_q.py \
  --episodes 8000 --alpha 0.01 --alpha-decay 0.999 \
  --gamma 0.99 --epsilon 0.2 --eps-decay 0.995 \
  --normalize-features --reward winloss --random-openings 8 \
  --l2 1e-4 --clip-params 1.0 --checkpoint 1000
```

Tips:
- Corners are a subset of stable discs;学習で `stable_w` が正に育つとコーナー選好が強化されます。
- ランダム初手を増やすと中盤以降の局面バリエーションが増え、コーナー周りの学習が進みます。

Options:

- --normalize-features: normalize features to comparable scales
  - position: each of 16 quadrant features divided by 4
  - mobility: divided by `--mob-scale` (default 20)
  - stable: divided by `--stab-scale` (default 28)
- --reward {winloss,margin}: terminal reward either win/loss (+1/0/-1) or disc margin normalized by 64
- --alpha-decay: multiplicative LR decay per episode (default 1.0)
- --eps-decay: multiplicative epsilon decay per episode
- --random-openings N: add N random opening plies to diversify start states
- --checkpoint N: save intermediate weights every N episodes
- --init-weights PATH: initialize from a previous `weights.json`
- --clip-params X: clip each parameter to [-X, X]
- --l2 λ: L2正則化係数（weight decay）。更新は `w ← (1−αλ)w + α·TD·x`。

## Developer Tools

Using the Makefile for quick workflows:

- Run tests: `make test`
- Watch and auto-run tests on change: `make watch`
- Coverage (if `coverage` installed): `make coverage`
- Run game (requires pygame): `make run`
- Quick training (writes weights.json): `make train`
