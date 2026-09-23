# Backgammon

A transformer learns backgammon by reading move sequences. Training does not use search. The Pygame window plays a human against that model, and falls back to search (minimax with alpha-beta) when every neural suggestion is illegal.

## Train

```bash
pip install -r requirements-training.txt
source .venv/bin/activate
python Backgammon_Brain_9_22_26.py
```

Needs a GPU. Press Enter at each prompt to keep the default.

A new model asks for size, then optimizer (`adamw`) and scheduler (`plateau`). Defaults: 8 layers, 384 embedding, 8 query heads, 2 KV heads, sequence 512, dropout 0.1, batch 128, 5 epochs.

Loading a checkpoint asks for batch size, epochs, learning rate, weight decay, and dropout. The epoch default on resume is 20, not 5. When it says the checkpoint was trained without frequency weighting, press Enter (`n`). `y` on that checkpoint does not build usable weights.

Checkpoints go in `/home/jonathan/Data/Backgammon_Model_*` (on a Mac, `/Users/jonathanrothberg/Data/...`). A line of loss, a sample, and a `.pth` are written every 100 batches. The first compiled step is quiet; the GPU being busy means that step is running. `BACKGAMMON_NO_COMPILE=1` skips `torch.compile`.

What the loss uses:

- Games are `<STARTGAME> <W|B> moves… <EOFG>`. Every training sequence starts at `<STARTGAME>`.
- Checker moves by the side that lost count 0.5. The winner's moves count 1. Dice and special tokens are not loss targets.
- A value head predicts White win / Black win. On half the games the input result is replaced by `<U>` so the head cannot read the winner off the prompt.
- Games with no result train the move picker only.

## Data

```bash
python generate_backgammon_games.py 1000 ./backgammon_games
python Backgammon_SGF_to_TXT.py
```

`generate_backgammon_games.py` needs GNU Backgammon (`gnubg`) and writes `.sgf` files. `Backgammon_SGF_to_TXT.py` turns a folder of those files into one `.txt` for the trainer. `setup_backgammon_games.sh` is the macOS installer plus a generator.

A labeled game looks like:

```
<STARTGAME> <W> <1B> d6 d1 m_ab m_cd <EOM> ... <EOFG>
```

`<W>` is who won. `<1B>` is who moved first. The trainer uses `<1B>` only to weight loser moves; it is not a model token. `<U>` is not in the file. Training inserts it on some inputs.

| Token | Meaning |
|-------|---------|
| `d1`-`d6` | One die |
| `m_xy` | One checker, point x to point y |
| `<EOM>` | End of this player's turn |
| `<NOMOVE>` | No legal move |
| `<STARTGAME>` / `<EOFG>` | Game boundaries |
| `<W>` / `<B>` | White or Black won |
| `<U>` | Result hidden (training and the win-guesser) |
| `<PAD>` | Fills a short game out to the sequence length |

Points are SGF letters: `a`-`x` = points 1-24, `y` = bar, `z` = bear off.

## Play

```bash
pip install -r requirements.txt
python Backgammon_9_22_26.py
```

You are the blue pieces (White in the model). The AI is pink (Black). `Backgammon_Inference.py` must sit next to the game, and you pick a `.pth` when the window starts.

- **Space**: opening roll, and every later roll on your turn. A tie on the opening roll needs Space again.
- **Click a point**: select one of your checkers, then click the destination. Click the same point to deselect.
- **Bar**: click the vertical line in the center, not the checker. Your bar checkers are drawn to the left of that line, outside the click strip.
- **Bear off**: click that same center line once you have nothing on the bar.
- **v**: AI vs AI
- **r** / **q**: restart / quit after a game

If the model's first choice is illegal, the terminal prints `LLM 1st choice was not legal` immediately. A later legal turn can replace that pick when the win-guess is better. At the end, the window shows how often the first choice was legal and how often the win-guesser overrode it. Search runs only when no suggestion is legal.

## Plot the loss

```bash
python plot_loss.py
python plot_loss.py /home/jonathan/Data/Backgammon_Model_B8H8E384K2
```

Loss is read from the checkpoint filename (`_L0.835_`). The argument is the folder the file dialog opens in.

## Files

| File | Use |
|------|-----|
| `Backgammon_Brain_9_22_26.py` | Train |
| `Backgammon_Inference.py` | Load a checkpoint and pick a move |
| `Backgammon_9_22_26.py` | Play |
| `Backgammon_SGF_to_TXT.py` | `.sgf` folder to one training `.txt` |
| `generate_backgammon_games.py` | Headless GNU Backgammon games |
| `setup_backgammon_games.sh` | macOS: install GNU Backgammon and generate games |
| `plot_loss.py` | Loss curve |
| `requirements.txt` | Packages to play |
| `requirements-training.txt` | Packages to train and plot |

## Troubleshooting

**No neural moves / falls back to search:** `Backgammon_Inference.py` must sit next to the game, and a `.pth` must be loaded.

**First choice is often illegal:** the checkpoint is not matching the board. The end-of-game counts show that.

**Bar click does nothing:** click the center line. Clicking the checker itself hits a board point.

**Training looks stuck after the first batch:** loss prints every 100 batches, and the first `torch.compile` step is slow. Out of memory: lower the batch size when the trainer asks.
