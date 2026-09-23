# Backgammon Game Implementation

A complete backgammon implementation with traditional AI and LLM-powered AI, featuring standard SGF notation and comprehensive game state tracking.

## 🎯 Overview

This is a fully functional backgammon game with:
- **Traditional AI**: Minimax algorithm with position evaluation
- **LLM AI**: Neural network model trained on expert games
- **Standard Notation**: Uses official backgammon SGF format
- **Game History**: Complete move tracking for AI training

## 🎲 Game Rules

Standard backgammon rules apply:
- 24 positions on the board
- 2 players (White moves counter-clockwise, Black moves clockwise)
- Pieces start at opposite ends and move toward their home boards
- Hit pieces go to the bar and must re-enter before other moves
- Bear off pieces from home board when all pieces are there

## 📍 Board Representation

### Internal Position Mapping
```
Board positions 1-24 correspond directly to labels a-x:
Position 1 (bottom right) = 'a'
Position 2 = 'b'
...
Position 24 (top right) = 'x'

Special positions:
- White bar = 25 (internal)
- Black bar = -1 (internal)
- White bearing off = 0 (internal)
- Black bearing off = 25 (internal)
```

### Visual Layout
```
Top Row (positions 13-24): 24(x) ← 13(m)
Bottom Row (positions 1-12): 1(a) → 12(l)
```

## 🔤 Standard SGF Nomenclature

The game uses official backgammon SGF notation:

### Board Positions
- **`a-x`**: Positions 1-24 on the board
  - `a` = position 1 (bottom right)
  - `x` = position 24 (top right)

### Special Positions
- **`y`**: Bar (where pieces go when hit, shared by both players)
- **`z`**: Bearing off (moving pieces off the board)

### Move Examples
```
d42 m_qusu     # Roll 4,2; move q→u→s→u
d51 m_yvid     # Roll 5,1; move bar→v→i→d
d32 m_rxxz     # Roll 3,2; move r→x→x→off
```

### Move Sequence Format
Moves are recorded as: `m_` + sequence of position letters
- Each pair represents: start_position → end_position
- Multiple moves in one turn are concatenated
- Bearing off uses `z`, bar entry uses `y`

## 🤖 AI Implementation

### Search AI (Traditional)
- Uses minimax algorithm with position evaluation
- Evaluates board state based on:
  - Pip count (distance to bearing off)
  - Blot vulnerability (single pieces)
  - Home board control
  - Prime formation
  - Attacking potential

### LLM AI
- Trained on expert backgammon games
- Uses game history context for strategic decisions
- Parses moves in standard notation
- Learns from board state changes (including implicit hitting)

## 📁 File Structure

```
backgammon_Nov_1_25.py          # Main game implementation
backgammon_games_20251031_181006.txt  # Training game database
Backgammon_README.md            # This documentation
```

## 🎮 Game Flow

1. **Dice Roll**: Random 1-6 for each die
2. **Move Selection**: Player chooses legal moves using dice values
3. **Validation**: Moves checked against backgammon rules
4. **Execution**: Pieces move, hits send opponents to bar
5. **Turn End**: Game state recorded for AI learning

## 🔧 Technical Details

### Key Functions
- `is_valid_move()`: Validates moves according to rules
- `perform_search_ai_with_move_capture()`: Traditional AI decision making
- `evaluate_board()`: Position evaluation for AI
- `idx_to_letter()` / `letter_to_idx()`: Position notation conversion

### Data Structures
- `board`: List of 25 integers (index 0 unused, 1-24 = positions)
- `bar`: List of 2 integers [white_pieces, black_pieces]
- `white_pieces_off` / `black_pieces_off`: Bearing off counters

### AI Training Data
Games are recorded in standard SGF format with full move sequences, enabling LLM learning of:
- Opening strategies
- Midgame tactics
- Endgame technique
- Hit avoidance and execution

## 🚀 Usage

Run the game:
```bash
python3 backgammon_Nov_1_25.py
```

Game modes:
- **Human vs AI**: Play against computer
- **Versus Mode**: AI vs AI for training data
- **LLM Mode**: Use trained neural network AI

## 📊 Training Data Format

Each game in the database follows:
```
<STARTGAME> d[die1][die2] m_[moves] <EOFG>
```

Example:
```
<STARTGAME> d42 m_qusu d51 m_yvid d32 m_rxxz <EOFG>
```

This format allows the LLM to learn complete games and understand strategic patterns.

## 🎯 Development Notes

- **Position Mapping**: Internal positions 1-24 match external labels a-x exactly
- **Standard Compliance**: Uses official backgammon SGF notation
- **Extensible AI**: Easy to add new AI implementations
- **Complete State Tracking**: All game events recorded for analysis

---

*This implementation maintains full compatibility with standard backgammon notation and game databases.*
