# Backgammon

A transformer learns backgammon by reading move sequences. Training does not use a search engine. The Pygame window can play a human, that neural model, or a separate search on the AI side.

When the neural net's suggestions are illegal, the game falls back to search (minimax with alpha-beta pruning).

## Train

```bash
pip install -r requirements-training.txt
source .venv/bin/activate          # if you use a project venv
python Backgammon_Brain_9_22_26.py
```

The script asks for a new model or a checkpoint, then a games file (`.txt`), then size and batch. It needs a GPU.

What the trainer learns from the result:

- Each game is `<STARTGAME> <W|B> moves… <EOFG>`. `<W>` / `<B>` are the GNU Backgammon colors (the same letters as `;W` / `;B` in the SGF).
- Checker moves by the side that lost are weighted 0.5. The winner's moves stay 1.0. Turns alternate from the first `;W` or `;B` in that game.
- A value head predicts White win / Black win. In half the games the input result token is replaced by `<U>` so the head cannot read the answer off the prompt.
- Games with no result in the file stay unlabeled. They train the move picker only.

Older checkpoints still load. `<W>`, `<B>`, `<U>` and the value head are added when they are missing. An older optimizer state is kept when the parameter count still matches.

Every training sequence starts at `<STARTGAME>`, so the model sees the game from the opening roll.

## Data

```bash
./setup_backgammon_games.sh 1000
python generate_backgammon_games.py 1000 ./backgammon_games
python Backgammon_SGF_to_TXT.py
```

`setup_backgammon_games.sh` installs GNU Backgammon and can generate games. `generate_backgammon_games.py` writes `.sgf` files. `Backgammon_SGF_to_TXT.py` turns a folder of those files into one `.txt` for the trainer.

A labeled game looks like:

```
<STARTGAME> <W> <1B> d6 d1 m_ab m_cd <EOM> ... <EOFG>
```

`<W>` is who won. `<1B>` means Black moved first (the token is recorded for training weights and is not a model token). `<U>` is not in the file. Training inserts it on some inputs.

| Token | Meaning |
|-------|---------|
| `d1`-`d6` | One die |
| `m_xy` | Move from point x to point y |
| `<EOM>` | End of this player's turn |
| `<NOMOVE>` | No legal move |
| `<STARTGAME>` / `<EOFG>` | Game boundaries |
| `<W>` / `<B>` | White or Black won |
| `<U>` | Result hidden (training and win-guesser only) |
| `<PAD>` | Fills a short game out to the sequence length |

Points are SGF letters: `a`-`x` = points 1-24, `y` = bar, `z` = bear off.

## Play

```bash
pip install -r requirements.txt
python Backgammon_9_22_26.py
```

- **Left click**: select and move
- **Right click**: deselect
- **Dice area**: click to roll
- **Space**: roll, including the opening roll
- **v**: AI vs AI
- **r** / **q**: restart / quit after a game

The neural net only suggests turns. The game throws out illegal ones. If its first choice is illegal, that is printed immediately (`LLM 1st choice was not legal`) so a weak checkpoint is obvious. If the checkpoint has a value head, a later legal turn can replace the first legal pick when the win-guess is better. At the end of the game the window reports how often the first choice was legal and how often the win-guesser overrode it.

Search does not call the neural net. It is the fallback when no suggestion is legal.

## Plot the loss

```bash
python plot_loss.py
python plot_loss.py /home/jonathan/Data/Backgammon_Model_B8H8E384K2
```

Loss is read from the checkpoint filename (`_L0.835_`).

## Files in this folder

| File | Use |
|------|-----|
| `Backgammon_Brain_9_22_26.py` | Train |
| `Backgammon_Inference.py` | Load a checkpoint and pick a move |
| `Backgammon_9_22_26.py` | Play |
| `Backgammon_SGF_to_TXT.py` | `.sgf` folder to one training `.txt` |
| `generate_backgammon_games.py` | Headless GNU Backgammon games |
| `setup_backgammon_games.sh` | Install GNU Backgammon and generate games |
| `plot_loss.py` | Loss curve |
| `requirements.txt` | Packages to play |
| `requirements-training.txt` | Packages to train and plot |
| `legacy/` | Older converter, notes, and the deprecated requirements file |

## Model

- Grouped-query attention, RMSNorm, SwiGLU
- Attention does not cross `<STARTGAME>` boundaries
- Default: 8 layers, 384 embedding, 512 tokens

## Troubleshooting

**No neural moves / falls back to search:** `Backgammon_Inference.py` must sit next to the game, and a `.pth` must be loaded.

**First choice is often illegal:** the checkpoint is not matching the board. The end-of-game counts are there to show that.

**Training out of memory:** lower the batch size when the trainer asks.
