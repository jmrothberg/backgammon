# Backgammon with LLM-Powered AI

A sophisticated backgammon game implementation featuring both traditional search-based AI and cutting-edge large language model (LLM) integration for intelligent move prediction.

## 🎯 Features

- **Human vs AI Gameplay**: Play against intelligent AI opponents
- **Dual AI Systems**:
  - **LLM-Powered AI**: Uses transformer models trained on backgammon games for strategic play
  - **Traditional Search AI**: Fallback system using minimax with alpha-beta pruning
- **Complete Game State Tracking**: Maintains full game history for LLM context
- **Debug-Friendly Architecture**: Comprehensive logging and error handling
- **Atomic Tokenization**: Advanced move representation system
- **Automatic Fallback**: Seamlessly switches from LLM to search AI when needed

## 🚀 Quick Start

### Prerequisites

**For Playing the Game:**
- Python 3.8+
- PyGame
- PyTorch (for LLM inference)
- NumPy

**For Training AI Models:**
- GNU Backgammon (http://www.gnu.org/software/gnubg/) - for generating expert-level training games
- Large dataset of SGF format backgammon games
- Significant computational resources (GPU recommended for training)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/jmrothberg/backgammon.git
cd backgammon
```

2. Install dependencies:
```bash
pip install pygame torch numpy
```

3. Run the game:
```bash
python backgammon_Atom_Dec_11_25.py
```

## 🎮 How to Play

### Basic Rules
- **Objective**: Be the first to bear off all your pieces
- **Movement**: Roll dice and move pieces along the board
- **Hitting**: Land on opponent's single piece to send it to the bar
- **Bearing Off**: Remove pieces from your home board when all pieces are there

### Controls
- **Left Click**: Select and move pieces
- **Right Click**: Deselect piece
- **Roll Dice**: Click the dice area when it's your turn
- **Auto-Roll**: Toggle automatic dice rolling in settings

### AI Opponents
- **LLM AI**: Uses trained transformer models for strategic play (requires model file `.pth`)
- **Search AI**: Traditional AI using position evaluation (always available as fallback)

The game automatically falls back to Search AI when:
- No model file is loaded
- LLM prediction fails to parse
- LLM suggests illegal moves

## 🧠 AI Architecture

### Transformer Model
The backgammon AI uses a custom transformer architecture optimized for game move prediction:

- **MultiQueryAttention (GQA)**: Memory-efficient attention with grouped query heads
- **RMSNorm**: Faster normalization that improves training stability
- **SwiGLU Activation**: Better than standard ReLU/GELU for language models
- **Game Boundary Masking**: Prevents attention across different games in training batches
- **Autoregressive Generation**: Predicts next move token given full game context

### Tokenization System (Atomic Format)
- **Dice Tokens**: `d1` through `d6` for individual dice values
- **Move Tokens**: `m_xy` pairs where x=source position, y=destination (using SGF notation a-x)
- **Special Tokens**: `<STARTGAME>`, `<EOFG>`, `<EOM>` (end of move), `<NOMOVE>`, `<PAD>`
- **Position Notation**: a-x = board positions 1-24, y = bar, z = bearing off

### LLM Integration
The game integrates with `BackgammonMovePredictor_Standalone_Atom.py`, which provides:
- **Device-Aware Inference**: Automatically uses GPU/CPU/MPS based on availability
- **Atomic Tokenization**: Efficient move representation with paired move tokens
- **Context-Aware Prediction**: Uses full game history for strategic decision making
- **Beam Search**: For doubles (4 moves), uses beam search to find optimal move sequences

### Training Data
The project includes tools for generating and processing backgammon game data:
- `generate_backgammon_games.py`: Creates training games
- `convert_old_to_atomic_format.py`: Processes game data for training
- SGF format game databases (excluded from repo due to size)

### Training the AI Model
- `BackgammonBrain_Atom_Pergame_11_20_25.py`: Complete transformer training implementation
  - Trains from expert GNU Backgammon games
  - Uses per-game training strategy for better learning
  - Produces models that can be used with the inference engine

## 🏗️ Training Prerequisites & Data Generation

### GNU Backgammon Setup
To train the AI model, you need expert-level backgammon games. The best source is GNU Backgammon:

1. **Download GNU Backgammon**:
   - Visit: http://www.gnu.org/software/gnubg/
   - Download the appropriate version for your operating system
   - Install the program

2. **Generate Training Games**:
   ```bash
   # Run GNU Backgammon and use its analysis features to generate games
   # Save games in SGF format (Smart Game Format)
   ```

### Using the Game Generator Script
The included `generate_backgammon_games.py` is a wrapper that automates GNU Backgammon to generate training games:

```bash
# Generate 1000 games in the output directory
python generate_backgammon_games.py 1000 ./backgammon_games

# With instance ID and worker count for parallel generation
python generate_backgammon_games.py 1000 ./backgammon_games 0 32
```

**Important**: This script requires GNU Backgammon (`gnubg`) to be installed and accessible from the command line. It runs gnubg in headless mode to generate expert-level games.

### Data Processing Pipeline
1. **Install GNU Backgammon**: Download from http://www.gnu.org/software/gnubg/
2. **Generate SGF Games**: Use `generate_backgammon_games.py` to create expert-level games
3. **Convert to Atomic Format**: Use `convert_old_to_atomic_format.py` to tokenize game data
   - Splits dice tokens: `d41` → `d4 d1`
   - Splits move tokens: `m_lpab` → `m_lp m_ab`
   - Adds turn markers: `<EOM>` after each turn
4. **Train Model**: Run `BackgammonBrain_Atom_Pergame_11_20_25.py` on the processed data
5. **Deploy**: Use trained model with `BackgammonMovePredictor_Standalone_Atom.py`

### Training Data Requirements
- **Quality**: Expert-level games preferred (GNU Backgammon analysis)
- **Quantity**: Thousands of games for meaningful training
- **Format**: SGF files (standard backgammon game format)
- **Processing**: Convert to atomic tokenization format before training

### Model Hyperparameters
Default training configuration (can be adjusted in `BackgammonBrain_Atom_Pergame_11_20_25.py`):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_embd` | 384 | Embedding dimension |
| `n_head` | 8 | Number of attention heads |
| `n_kv_heads` | 2 | Key-value heads (GQA) |
| `n_layer` | 8 | Transformer layers |
| `block_size` | 512 | Max sequence length |
| `dropout` | 0.1 | Dropout rate |
| `batch_size` | 128 | Training batch size |
| `learning_rate` | 4e-4 | Initial learning rate |

## 🛠️ Project Structure

```
backgammon/
├── backgammon_Atom_Dec_11_25.py          # Main game file
├── BackgammonBrain_Atom_Pergame_11_20_25.py    # Transformer training script
├── BackgammonMovePredictor_Standalone_Atom.py  # LLM inference engine
├── generate_backgammon_games.py          # Game data generation
├── convert_old_to_atomic_format.py       # Data processing
├── README_BACKGAMMON_PER_GAME.md         # Training strategy docs
├── .gitignore                           # Excludes data files
└── README.md                            # This file
```

## 🔧 Configuration

### LLM Setup
The game automatically detects and uses the LLM predictor if available. To disable LLM features, modify the import section in `backgammon_Atom_Dec_11_25.py`.

### Debug Mode
Enable detailed logging by checking terminal output for:
- ✅ Success indicators
- ❌ Error messages
- 🔄 AI thinking processes
- ⚠️ Warning messages

## 🐛 Troubleshooting

### Common Issues

1. **LLM Not Available**
   - Ensure `BackgammonMovePredictor_Standalone_Atom.py` is in the same directory
   - Check PyTorch installation
   - Game falls back to search AI automatically

2. **Game Freezes**
   - Check dice validation in `is_valid_move()` function
   - Verify board state integrity
   - Look for "board[:]" copying issues in logs

3. **Invalid Moves**
   - Ensure dice match available moves
   - Check board position calculations
   - Verify move token parsing

### Debug Workflow
1. Check terminal output for emoji indicators
2. Look for "LLM_AI thinking" vs "Search AI evaluating"
3. Verify `game_history_tokens` format
4. Check board state with debug prints if needed

## 📚 Documentation

- `README_BACKGAMMON_PER_GAME.md`: Detailed explanation of the per-game training strategy
- Code comments include comprehensive debugging guides
- Inline documentation covers key algorithms and data structures

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly (both LLM and search AI modes)
5. Submit a pull request

## 📄 License

This project is open source. Please check individual files for license information.

## 🙏 Acknowledgments

- Built with PyGame for graphics
- LLM components use PyTorch
- Training data generated using custom game engines
- Special thanks to the backgammon community for game databases