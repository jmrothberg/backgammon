# Backgammon with LLM-Powered AI

A backgammon game where you play against a transformer neural network trained on expert games. The AI learns move patterns from thousands of GNU Backgammon games and predicts moves autoregressively, like a language model predicts words.

When the neural network can't find a legal move, the game falls back to a traditional search-based AI (minimax with alpha-beta pruning).

## Dependencies

| What you want | Install command |
|-----------------|-----------------|
| **Play** (Pygame UI + PyTorch to run the trained model) | `pip install -r requirements.txt` |
| **Train** the model, or use plotting / training-adjacent scripts | `pip install -r requirements-training.txt` |

`requirements-training.txt` includes everything from `requirements.txt` and adds `transformers` (training optimizer) and `matplotlib` (e.g. plotting loss from checkpoints).

Recommended: use a virtual environment so this project does not affect your system Python.

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
```

**Note:** `Backgammon_requirements.txt` is deprecated; pins live only in `requirements.txt` and `requirements-training.txt`.

## File Guide

### Playing the Game

| File | What it does |
|------|-------------|
| `backgammon_Atom_Dec_11_25.py` | **The game.** Pygame UI, board rendering, game rules, human vs AI play. Run this to play. Imports the inference engine for AI moves. |
| `BackgammonMovePredictor_Standalone_Atom.py` | **Inference engine.** Loads a trained `.pth` model and predicts moves given the game history and a dice roll. The game imports `BackgammonMovePredictor` from this file. Contains a copy of the model architecture (so it can reconstruct the network from saved weights) plus the prediction logic: joint-probability search for normal rolls (2 moves), beam search for doubles (4 moves). Called "Standalone" because it works without importing the training script. |
| `requirements.txt` | Minimal Python dependencies for playing (see [Dependencies](#dependencies)). |
| `requirements-training.txt` | Everything above plus packages for training and optional tooling (see [Dependencies](#dependencies)). |

### Training the AI

| File | What it does |
|------|-------------|
| `BackgammonBrain_Atom_Pergame_11_20_25.py` | **Training script.** Defines the transformer architecture (Grouped Query Attention, RMSNorm, SwiGLU), loads tokenized game data, and trains the model. Uses a "per-game" strategy where every training sequence starts at `<STARTGAME>` so the model always sees full game context. Outputs `.pth` checkpoint files. Requires a GPU. |
| `generate_backgammon_games.py` | **Data generation.** Runs GNU Backgammon (`gnubg`) in headless mode to produce expert-level games in SGF format. Supports parallel generation across multiple CPU cores. |
| `setup_backgammon_games.sh` | **Setup helper.** Shell script that installs GNU Backgammon (via Homebrew on macOS or apt on Linux) and then runs `generate_backgammon_games.py` to produce training games. |
| `convert_old_to_atomic_format.py` | **Data converter.** Converts older combined-token format to the current atomic format: splits `d41` into `d4 d1`, splits `m_lpab` into `m_lp m_ab`, and adds `<EOM>` turn markers. |
| `README_BACKGAMMON_PER_GAME.md` | Explains why the per-game training strategy works better than the old overlapping-window approach. |
| `Backgammon_SGF_to_TXT_Converter_Atom.py` | Optional: GUI tool — folder of `.sgf` games → one atomic-token `.txt` for training pipelines. |
| `plot_loss_Nov_9_25.py` | Optional: scan a checkpoint folder and plot loss vs training step from filenames. |

### Other

| File | What it does |
|------|-------------|
| `.gitignore` | Excludes model checkpoints (`.pth`), game data (`.sgf`), virtual environments, and OS files from git. |

## Documentation

| Document | Purpose |
|----------|---------|
| `README.md` (this file) | Setup, file map, quick start, tokenization overview. |
| `README_BACKGAMMON_PER_GAME.md` | Why per-game training sequences beat sliding windows. |
| `Backgammon_README.md` | Longer reference: rules, board mapping, SGF notation. |
| `README_BackgammonMovePredictor_Standalone.md` | Inference-only API, GPU device mapping, loading models on different hardware. |
| `ATOMIC_TOKENIZATION_IMPLEMENTATION_GUIDE.md` | Design and migration notes for atomic tokenization. |

## How the Pieces Fit Together

```
1. GENERATE DATA        2. CONVERT FORMAT       3. TRAIN MODEL           4. PLAY

setup_backgammon_       convert_old_to_         BackgammonBrain_         backgammon_Atom_
games.sh                atomic_format.py        Atom_Pergame.py          Dec_11_25.py
    |                       |                       |                       |
    v                       v                       v                       v
Install gnubg +         .sgf/.txt games  -->    Atomic token     -->    Trained .pth    -->  Pygame UI
generate_backgammon_    "d41 m_lpab"            files                    model file           (human plays
games.py                    |                   "d4 d1 m_lp m_ab"           |                 against AI)
    |                       v                       |                       |
    v                   Atomic format               v                       v
Expert .sgf games       training data           .pth checkpoint     BackgammonMovePredictor_
                                                                    Standalone_Atom.py
                                                                    (loads model, predicts moves)
```

## Quick Start

### Play the Game

```bash
git clone https://github.com/jmrothberg/backgammon.git
cd backgammon
python3 -m venv .venv && source .venv/bin/activate   # optional but recommended
pip install -r requirements.txt
python backgammon_Atom_Dec_11_25.py
```

The game will prompt you to select a `.pth` model file. If no model is available, it uses the search-based AI.

### Controls

- **Left Click**: Select and move pieces
- **Right Click**: Deselect
- **Dice Area**: Click to roll

### Train Your Own Model

1. Install GNU Backgammon and generate training games:
   ```bash
   ./setup_backgammon_games.sh 1000
   ```
   Or manually:
   ```bash
   # macOS
   brew install gnubg
   # Linux
   sudo apt-get install gnubg

   python generate_backgammon_games.py 1000 ./backgammon_games
   ```

2. Convert to atomic token format:
   ```bash
   python convert_old_to_atomic_format.py
   ```

3. Train (requires GPU):
   ```bash
   pip install -r requirements-training.txt
   python BackgammonBrain_Atom_Pergame_11_20_25.py
   ```

4. Play with your trained model:
   ```bash
   python backgammon_Atom_Dec_11_25.py
   ```

## Tokenization

The AI represents games as token sequences using "atomic" format:

| Token | Meaning | Example |
|-------|---------|---------|
| `d1`-`d6` | Individual die value | `d5 d2` = rolled 5 and 2 |
| `m_xy` | Move from position x to position y | `m_lp` = move from position l to p |
| `<EOM>` | End of turn | Separates one player's moves from the next |
| `<STARTGAME>` | Beginning of a game | Always the first token |
| `<EOFG>` | End of game | Always the last token |
| `<NOMOVE>` | No legal move available | Player must pass |
| `<PAD>` | Padding | Fills short sequences to fixed length during training |

Position notation uses SGF-style letters: `a`-`x` = board positions 1-24, `y` = bar, `z` = bear off.

## Model Architecture

The transformer uses techniques from modern LLMs, scaled down for backgammon:

- **Grouped Query Attention (GQA)**: 8 query heads, 2 key-value heads (memory efficient)
- **RMSNorm**: Faster and more stable than LayerNorm
- **SwiGLU**: Better activation function than ReLU/GELU
- **Game Boundary Masking**: Prevents attention across different games in training batches
- Default: 8 layers, 384 embedding dim, 512 max sequence length (~35M parameters)

## Troubleshooting

**No LLM moves / falls back to Search AI**: Make sure `BackgammonMovePredictor_Standalone_Atom.py` is in the same directory as the game and that you have a `.pth` model file loaded.

**Game freezes**: Check terminal output for error messages. The game logs all AI decisions with indicator prefixes.

**Training out of memory**: Reduce `batch_size` in `BackgammonBrain_Atom_Pergame_11_20_25.py`.
