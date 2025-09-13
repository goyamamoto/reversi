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
- A: toggle simple AI for White (Black is human)
- ESC or Q: quit

## Rules

- Standard 8x8 Reversi rules
- If current player has no valid moves, the turn automatically passes
- Game ends when neither player has a valid move or the board is full

## Notes

- The built-in AI is simple (greedy + corners first) and intended for casual play.

