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

## Weights Versioning & Auto-Training

- Versioned saving: `train_q.py` に `--auto-version` と `--save-dir weights` を指定すると、`weights/weights-YYYYMMDD-HHMMSS-<episodes>ep[-tag].json` に保存し、`weights/latest.json` を更新します。既定で `./weights.json` も更新します。
- Resume latest: `--resume-latest` で `weights/` の最新ファイルから継続学習します（`--init-weights` 未指定時）。
- Tagging: `--tag corner` などでファイル名にタグ付与。
- Metadata: 出力JSONのトップに `__meta__` を付与（作成時刻、エピソード数、ハイパラ、正規化設定、初期重みパス等）。`main.py` は余分なキーを無視して読み込みます。

Examples:

- Continue from latest and save versioned weights:
```
make train-auto
```

- Manual:
```
python train_q.py --resume-latest --auto-version --save-dir weights --out weights.json \
  --episodes 2000 --alpha 0.005 --alpha-decay 0.999 --gamma 0.99 \
  --epsilon 0.2 --eps-decay 0.995 --normalize-features --reward winloss \
  --random-openings 8 --l2 1e-4 --clip-params 1.0 --checkpoint 500 --tag v2
```

At runtime, the game loads weights in this order: `weights/latest.json` → `weights.json` → newest file under `weights/`（存在するもの）。

## Versioning, Pause/Resume, and Milestones

- Where weights are saved:
  - Versioned: `weights/weights-YYYYMMDD-HHMMSS-<episodes>ep[-tag].json`
  - Pointer: `weights/latest.json`（最新学習結果）と `./weights.json`（ゲーム起動時既定）
  - Milestones: `weights/milestones/*.json`（到達点スナップショット）

- Pause/Resume training:
  - `make train-auto` は常に最新から継続し、終了時にバージョン保存＋latest更新
  - `make auto-tune` は複数サイクルの「学習→評価→監査→調整」を実行。途中で Ctrl+C で中断しても、その時点までの成果（latest/バージョン/チェックポイント）は残り、再実行で最新から継続
  - 手動再開: `python train_q.py --resume-latest --auto-version --save-dir weights --out weights.json ...`

- First milestone: Corner preference
  - Auto-tune に `--until-corner` を付与すると、角が合法の局面で角を選ぶ割合（corner preference rate）がしきい値（既定70%）に達した時点で停止
  - 達成時にコンソールへ「First milestone reached: ...」を表示し、同時に `weights/milestones/corner-*.json` として重み＋メタデータを保存
  - 角の機会が少ない場合は、学習中のランダム初手数を自動で増やして多様な局面を生成

## Evaluate & Auto-Tune

- Evaluate learned weights vs past versions (pool, deterministic order):
```
python eval_agents.py --matches-per-opp 10 --depth 4 --random-openings 0 \
  --opponent pool --opp-dir weights --opp-order newest --opp-limit 5 --progress
```
Outputs JSON like `{ "games": 50, "wins": 31, "losses": 17, "draws": 2, "win_rate": 0.62 }`.

Notes:
- `--matches-per-opp` を使うと、プール中の各相手と同数の試合を行い、相手選択は指定順序でラウンドロビンになります。
- 再現性重視なら `--random-openings 0` と固定の探索深さを使ってください。

- Auto-tuner (train → evaluate →調整→繰り返し):
```
python auto_tuner.py --cycles 5 --episodes 1000 --normalize-features --until-corner --corner-threshold 0.7 --progress \
  --alpha 0.01 --alpha-decay 0.999 --epsilon 0.2 --eps-decay 0.995 \
  --l2 1e-4 --clip-params 1.0 --random-openings-train 6 \
  --depth 4 --matches 40 --random-openings-eval 6
```
`--progress` を付けると、各サイクルで「学習開始→評価（各試合進捗）→コーナー監査（各試合進捗）」のログが出力されます。
学習で改善が見られない場合は、学習率を下げ、探索率やL2を少し上げて再トライします。`weights/` にバージョン保存しつつ、`weights/latest.json` と `weights.json` を更新します。

Corner milestone behavior:
- `--until-corner` を付けると、角が打てる局面で角を選ぶ割合（corner preference rate）を監視し、しきい値到達時にマイルストーンを保存します。
- さらに `--continue-after-corner` を付けると、到達後も学習を継続します（停止せず、マイルストーンのみ記録）。
- 角の機会が少ない場合は、学習中の `--random-openings-train` を自動的に増やして多様な局面を生成します。
- 角選好が停滞している場合は、学習率を下げ、探索率とL2（正則化）を上げ、必要に応じて1サイクルのエピソード数を増やします。

Makefile shortcuts:
- `make eval`（進捗を逐次表示、対戦相手は weights/ の過去版サンプル） / `make auto-tune`

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

- Run tests: `make test` (excludes integration tests by default)
- Run all tests (incl. integration): `make test-all`
- Watch and auto-run tests on change: `make watch`
- Coverage (if `coverage` installed): `make coverage`
- Run game (requires pygame): `make run`
- Quick training (writes weights.json): `make train`
