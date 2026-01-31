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
- **LLM AI**: Uses trained language models for strategic play (requires model file)
- **Search AI**: Traditional AI using game tree search (always available)

## 🧠 AI Architecture

### LLM Integration
The game integrates with `BackgammonMovePredictor_Standalone_Atom.py`, which provides:
- **Device-Aware Inference**: Automatically uses GPU/CPU based on availability
- **Atomic Tokenization**: Efficient move representation
- **Context-Aware Prediction**: Uses full game history for decision making

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

### Alternative: Use Included Game Generator
If GNU Backgammon is unavailable, use the included Python game generator:

```bash
python generate_backgammon_games.py
```

**Note**: GNU Backgammon games provide much higher quality training data than synthetic generation.

### Data Processing Pipeline
1. **Generate/Aquire SGF Games**: Expert-level backgammon games in SGF format
2. **Convert to Atomic Format**: Use `convert_old_to_atomic_format.py` to process the data
3. **Train Model**: Run `BackgammonBrain_Atom_Pergame_11_20_25.py` on the processed data
4. **Deploy**: Use trained model with `BackgammonMovePredictor_Standalone_Atom.py`

### Training Data Requirements
- **Quality**: Expert-level games preferred (GNU Backgammon analysis)
- **Quantity**: Thousands of games for meaningful training
- **Format**: SGF files (standard backgammon game format)
- **Processing**: Convert to atomic tokenization format before training

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