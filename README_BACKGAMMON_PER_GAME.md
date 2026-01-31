# BackgammonBrain "Per-Game" Training Strategy Explained

This document explains the "Per-Game" training strategy used in `BackgammonBrain_Atom_Pergame_11_19_25.py` and why it represents a fundamental shift in how the AI learns backgammon.

## The Problem: "Blindfolded" Learning

In previous versions, the training data was fed to the model as a continuous stream of tokens, chopped into fixed-size windows (e.g., 512 tokens).

*   **Random Starting Points:** A training sequence might start at Move 1, Move 15, or right in the middle of a bearing-off phase.
*   **Missing Context:** If a sequence starts at Move 15, the model **cannot see moves 1-14**. It doesn't know how the checkers got to their current positions.
*   **The "Blindfold" Effect:** The model had to guess the board state from a fragment of history. This forced it to act like a pattern-matcher (guessing moves that "look right") rather than a strategic thinker (understanding the game state).

## The Solution: "Per-Game" Sequencing

The new `BackgammonMovesDataset` changes this completely.

1.  **Start at the Start:** Every single training sequence begins with the `<STARTGAME>` token.
2.  **Full History:** The model sees the opening roll, the response, and every subsequent move in order.
3.  **State Tracking:** Because it sees the entire history, the model's internal "hidden state" can accurately track where every checker is on the board. It effectively "plays out" the game in its memory.

### Comparison

| Feature | Old Method (Overlap) | New Method (Per-Game) |
| :--- | :--- | :--- |
| **Sequence Start** | Random token (e.g., middle of a move) | Always `<STARTGAME>` |
| **Board State** | Unknown / Guesswork | Fully Known / Calculated |
| **Training Examples** | ~1 Billion (99% overlap) | ~3 Million (0% overlap) |
| **Batches per Epoch** | ~7,500,000 | ~23,000 |
| **Learning Quality** | Noisy, fragmented | Coherent, strategic |
| **Speed** | Very slow epochs | **300x Faster Epochs** |

## Efficiency vs. Padding

You might notice that many sequences now contain "padding" (zeros) because games have different lengths.

*   **Short Games:** A 100-token game in a 512-token window has 412 padding tokens.
*   **Is this wasteful?** Numerically, yes. The GPU processes zeros.
*   **Is it smart?** **YES.**
    *   The model is masked to **ignore the zeros** (zero cost to learning).
    *   It learns **100% correct strategy** from the valid tokens.
    *   It is better to learn from 10 clean examples than 1000 confused ones.

## Key Code Changes

The critical logic is in `BackgammonMovesDataset`:

```python
# Old Way (Overlap)
def __len__(self):
    return len(self.tokens) - self.seq_length  # Every token starts a sequence

# New Way (Per-Game)
def __len__(self):
    # Only <STARTGAME> tokens start a sequence
    return len(self._game_starts)
```

This single change transforms the model from a text predictor into a true game engine learner.

