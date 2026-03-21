# Backgammon Atomic Tokenization Implementation Guide

## Table of Contents
1. [Overview](#overview)
2. [Tokenization Changes](#tokenization-changes)
3. [Special Tokens](#special-tokens)
4. [Converter Implementation](#converter-implementation)
5. [Training Implementation](#training-implementation)
6. [Inference Implementation](#inference-implementation)
7. [Game Engine Integration](#game-engine-integration)
8. [Edge Cases](#edge-cases)
9. [Testing Strategy](#testing-strategy)
10. [Benefits and Rationale](#benefits-and-rationale)
11. [Implementation Checklist](#implementation-checklist)

---

## Overview

### Problem Statement
Current tokenization uses compound tokens:
- Dice rolls: `d66`, `d41` (one token per roll)
- Moves: `m_abcd` (one token per complete move sequence)

This creates millions of unique tokens, preventing generalization to unseen combinations.

### Solution: Atomic Tokenization
Break everything into atomic units:
- **Dice**: Split `d66` → `d6` `d6` (individual dice)
- **Moves**: Split `m_abcd` → `m_ab` `m_cd` (pairs representing single piece movements)
- **Special tokens**: Add `<EOM>` (End of Move) and `<NOMOVE>` (no legal moves)

### Key Benefits
1. **Smaller vocabulary**: Dice 21→6 tokens, Moves thousands→~676 pairs
2. **Better generalization**: Model learns atomic patterns, composes unseen sequences
3. **Natural structure**: Each token = one die or one move (matches game mechanics)
4. **Joint probability**: Predicts coherent move sequences with strategic understanding

---

## Tokenization Changes

### Current Format
```
<STARTGAME> d66 m_abcd d41 m_xyzw <EOFG>
```

### New Format
```
<STARTGAME> d6 d6 m_ab m_cd m_ef m_gh <EOM> d4 d1 m_xy m_zw <EOM> <EOFG>
```

### Dice Tokenization Rules

**Current**: One token per dice roll
- `d66`, `d55`, `d44`, `d33`, `d22`, `d11` (doubles)
- `d65`, `d64`, `d63`, `d62`, `d61` (non-doubles)
- `d54`, `d53`, `d52`, `d51` (non-doubles)
- ... (21 total dice roll tokens)

**New**: Split into individual dice
- `d66` → `d6` `d6`
- `d41` → `d4` `d1`
- `d33` → `d3` `d3`
- `d12` → `d1` `d2`

**Dice vocabulary**: Only 6 tokens (`d1`, `d2`, `d3`, `d4`, `d5`, `d6`)

**Implementation**:
```python
def split_dice(dice_str):
    """
    Split dice roll string into individual dice tokens.
    
    Args:
        dice_str: Two-digit string like "66", "41", "33"
    
    Returns:
        List of dice tokens: ["d6", "d6"] or ["d4", "d1"]
    """
    if len(dice_str) == 2:
        die1 = dice_str[0]  # First die: "6" or "4"
        die2 = dice_str[1]  # Second die: "6" or "1"
        return [f"d{die1}", f"d{die2}"]
    else:
        # Handle unexpected formats
        return [f"d{dice_str}"]
```

### Move Tokenization Rules

**Current**: One token per complete move sequence
- `m_abcd` (entire sequence as one token)
- `m_abcdefgh` (entire sequence as one token)
- Creates thousands of unique tokens

**New**: Split into pairs (each pair = one piece movement)
- `m_abcd` → `m_ab` `m_cd` (2 moves)
- `m_abcdefgh` → `m_ab` `m_cd` `m_ef` `m_gh` (4 moves)
- `m_xz` → `m_xz` (1 move, already atomic)

**Move vocabulary**: ~676 possible pairs (26×26), fewer valid in practice

**Expected move counts**:
- **Doubles** (`d66`, `d55`, etc.) → 4 moves (8 characters)
- **Non-doubles** (`d41`, `d23`, etc.) → 2 moves (4 characters)

**Implementation**:
```python
def split_moves(move_str):
    """
    Split move sequence into atomic pair tokens.
    
    Args:
        move_str: Move sequence like "abcd", "abcdefgh", "xz"
    
    Returns:
        List of move pair tokens: ["m_ab", "m_cd"] or ["m_xz"]
    """
    tokens = []
    for i in range(0, len(move_str), 2):
        if i + 1 < len(move_str):
            # Normal case: pair of characters
            pair = move_str[i:i+2]
            tokens.append(f"m_{pair}")
        else:
            # Edge case: odd length (shouldn't happen in valid games)
            # Handle gracefully by treating single char as move
            tokens.append(f"m_{move_str[i]}")
    return tokens
```

---

## Special Tokens

### New Special Tokens

Add two new special tokens to vocabulary:

1. **`<EOM>`** (End of Move)
   - Marks end of move sequence for a turn
   - Inserted after all moves for a dice roll
   - Example: `d6 d6 m_ab m_cd m_ef m_gh <EOM>`

2. **`<NOMOVE>`** (No Move)
   - Indicates no legal moves available (player blocked)
   - Used when player cannot make any moves
   - Example: `d6 d6 <NOMOVE> <EOM>`

### Complete Special Token List

- `<STARTGAME>`: Start of game (existing)
- `<EOFG>`: End of game (existing)
- `<EOM>`: End of move sequence (NEW - within game)
- `<NOMOVE>`: No moves possible (NEW - within game)
- `<PAD>`: Padding token (existing, for batching)

### Special Token Usage Rules

1. **After dice + moves**: `d6 d6 m_ab m_cd <EOM>`
2. **After dice + no moves**: `d6 d6 <NOMOVE> <EOM>`
3. **After partial moves**: `d6 d6 m_ab m_cd <EOM>` (if only 2 moves possible)
4. **Game boundaries**: `<STARTGAME> ... <EOFG>` (unchanged)

---

## Converter Implementation

### File: `Backgammon_SGF_to_TXT_Converter.py`

### Changes to `parse_sgf_moves()` Function

**Current behavior**:
- Extracts `d66` or `d41` as single tokens
- Extracts `m_abcd` as single tokens
- Returns: `['d66', 'm_abcd', 'd41', 'm_xyzw']`

**New behavior needed**:

1. **Parse dice and split**:
   ```python
   # Input from SGF: "66" or "41"
   dice_tokens = split_dice(dice_str)  # ["d6", "d6"] or ["d4", "d1"]
   ```

2. **Parse moves and split**:
   ```python
   # Input from SGF: "abcd" or "abcdefgh"
   move_tokens = split_moves(move_str)  # ["m_ab", "m_cd"] or ["m_ab", "m_cd", "m_ef", "m_gh"]
   ```

3. **Determine expected move count**:
   ```python
   def is_doubles(dice_str):
       """Check if dice roll is doubles (11, 22, 33, 44, 55, 66)"""
       return len(dice_str) == 2 and dice_str[0] == dice_str[1]
   
   if is_doubles(dice_str):
       expected_moves = 4  # Doubles = 4 moves
   else:
       expected_moves = 2  # Non-doubles = 2 moves
   ```

4. **Add special tokens**:
   ```python
   # After moves: add <EOM>
   tokens.extend(move_tokens)
   tokens.append('<EOM>')
   
   # If no moves: add <NOMOVE> <EOM>
   if not move_tokens:
       tokens.append('<NOMOVE>')
       tokens.append('<EOM>')
   ```

### Complete Parser Logic Flow

For each turn in SGF (`;B[...]` or `;W[...]`):

```python
def parse_turn(match):
    """
    Parse a single turn from SGF format.
    
    Args:
        match: SGF move string like "66abcd" or "41xyzw"
    
    Returns:
        List of tokens: ["d6", "d6", "m_ab", "m_cd", "m_ef", "m_gh", "<EOM>"]
    """
    tokens = []
    
    # Step 1: Extract dice (first 2 characters)
    dice_str = match[:2]  # "66" or "41"
    
    # Step 2: Split dice into individual tokens
    dice_tokens = split_dice(dice_str)
    tokens.extend(dice_tokens)
    
    # Step 3: Extract moves (remaining characters)
    move_str = match[2:]  # "abcd" or "abcdefgh"
    
    # Step 4: Determine expected move count
    is_double = is_doubles(dice_str)
    expected_chars = 8 if is_double else 4
    
    # Step 5: Handle moves
    if move_str:
        # Split moves into pairs
        move_tokens = split_moves(move_str)
        tokens.extend(move_tokens)
        tokens.append('<EOM>')
    else:
        # No moves available (blocked)
        tokens.append('<NOMOVE>')
        tokens.append('<EOM>')
    
    return tokens
```

### Updated `parse_sgf_moves()` Function Structure

```python
def parse_sgf_moves(sgf_content):
    """
    Parse SGF content with atomic tokenization.
    
    Returns:
        List of atomic tokens: ['d6', 'd6', 'm_ab', 'm_cd', '<EOM>', ...]
    """
    tokens = []
    move_pattern = r';[BW]\[([^\]]+)\]'
    matches = re.findall(move_pattern, sgf_content)
    
    for match in matches:
        # Skip analysis data
        if not match or match.startswith('A[') or 'E ver' in match:
            continue
        
        # Pure dice roll (no move)
        if re.match(r'^\d+$', match):
            dice_tokens = split_dice(match)
            tokens.extend(dice_tokens)
            tokens.append('<NOMOVE>')
            tokens.append('<EOM>')
        
        # Dice + move: "66abcd" or "41xyzw"
        elif re.match(r'^\d+[a-zA-Z]+$', match):
            dice_str = match[:2]
            move_str = match[2:]
            
            # Split dice
            dice_tokens = split_dice(dice_str)
            tokens.extend(dice_tokens)
            
            # Split moves
            if move_str:
                move_tokens = split_moves(move_str)
                tokens.extend(move_tokens)
                tokens.append('<EOM>')
            else:
                tokens.append('<NOMOVE>')
                tokens.append('<EOM>')
        
        # Move without explicit dice (shouldn't happen, but handle)
        elif re.match(r'^[a-zA-Z]+$', match):
            move_tokens = split_moves(match)
            tokens.extend(move_tokens)
            tokens.append('<EOM>')
    
    return tokens
```

### Example Output Format

**Input SGF**:
```
;B[66abcdefgh]
;W[41xyzw]
;B[66]
```

**Output tokens**:
```
d6 d6 m_ab m_cd m_ef m_gh <EOM> d4 d1 m_xy m_zw <EOM> d6 d6 <NOMOVE> <EOM>
```

---

## Training Implementation

### File: `BackgammonBrain_Parallel_11_5_25.py`

### Vocabulary Changes

**Current vocabulary includes**:
- Dice tokens: `d11`, `d12`, ..., `d66` (21 tokens)
- Move tokens: `m_xxxx` (thousands of unique sequences)
- Special tokens: `<STARTGAME>`, `<EOFG>`, `<PAD>`

**New vocabulary will include**:
- Dice tokens: `d1`, `d2`, `d3`, `d4`, `d5`, `d6` (6 tokens)
- Move tokens: `m_xx` (pairs only, ~676 possible, fewer valid)
- Special tokens: `<STARTGAME>`, `<EOFG>`, `<EOM>`, `<NOMOVE>`, `<PAD>`

### Changes to `create_move_to_idx_from_text()`

**Current behavior**:
- Extracts all unique tokens from text
- Categorizes as dice (`d\d+`) or moves (`m_*`)

**New behavior needed**:
- Same extraction logic (should work automatically)
- Verify atomic tokens are created correctly
- Ensure `<EOM>` and `<NOMOVE>` are included

**No major changes needed** - function should work with new format automatically.

### Training Format (Unchanged)

Training remains **autoregressive** (next token prediction):

**Input sequence**:
```
d6 d6 m_ab m_cd m_ef m_gh <EOM> d4 d1 m_xy m_zw <EOM>
```

**Training targets** (shifted by one):
```
d6 m_ab m_cd m_ef m_gh <EOM> d4 d1 m_xy m_zw <EOM> <EOFG>
```

**Loss calculation** (standard autoregressive):
- `L(m_ab | d6, d6)` - predict first move
- `L(m_cd | d6, d6, m_ab)` - predict second move
- `L(m_ef | d6, d6, m_ab, m_cd)` - predict third move
- `L(m_gh | d6, d6, m_ab, m_cd, m_ef)` - predict fourth move
- `L(<EOM> | d6, d6, m_ab, m_cd, m_ef, m_gh)` - predict end

**No changes needed to training loop** - standard autoregressive training works perfectly.

### Dataset Changes

`BackgammonMovesDataset` should handle:
- Atomic dice tokens (`d6`, `d1`, etc.) - **no changes needed**
- Atomic move tokens (`m_ab`, `m_cd`, etc.) - **no changes needed**
- `<EOM>` tokens after each turn - **no changes needed**
- `<NOMOVE>` tokens when blocked - **no changes needed**

The dataset class tokenizes text by splitting on whitespace, so it will automatically handle the new format.

### Model Architecture (Minor Changes)

- Same transformer architecture
- Same attention mechanisms
- **Changed**: Context window (`block_size`) increased from 128 to 512
  - **Reason**: Atomic tokenization (splitting moves) creates longer sequences per game (approx 4x longer).
  - 128 tokens is no longer sufficient to see enough game history.
  - 512 ensures the model has adequate strategic context.
- Only vocabulary size changes (smaller!)

---

## Inference Implementation

### Current Inference

**Single token prediction**:
- Input: `d66`
- Output: Top-k single tokens (`m_abcd`, `m_xyzw`, etc.)
- Problem: Can't predict unseen combinations

### New Inference: Pair/Quadruple Prediction

**Goal**: Predict complete move sequences (pairs/quadruples) with joint probabilities.

### For Non-Doubles (d16, d41, etc.) - 2 Moves Needed

**Approach**: Get top-k pairs ranked by joint probability

```python
def get_top_k_pairs(model, dice_tokens, idx_to_move, move_to_idx, device, k=10):
    """
    Get top-k move pairs with joint probabilities for non-doubles.
    
    Args:
        model: Trained transformer model
        dice_tokens: List of dice tokens like ["d6", "d1"]
        idx_to_move: Token index to string mapping
        move_to_idx: Token string to index mapping
        device: PyTorch device
        k: Number of top pairs to return
    
    Returns:
        List of (joint_prob, [move1, move2]) tuples, sorted by probability
    """
    model.eval()
    
    # Convert dice tokens to indices
    context_indices = [move_to_idx[token] for token in dice_tokens if token in move_to_idx]
    context_tensor = torch.tensor([context_indices], dtype=torch.long).to(device)
    
    pairs = []
    
    with torch.no_grad():
        # Step 1: Get probability distribution for first move
        output, _ = model(context_tensor)
        first_logits = output[0, -1]  # Last position
        first_probs = torch.softmax(first_logits, dim=-1)
        
        # Get top candidates for first move (get more than k to ensure diversity)
        top_first_k = min(50, len(first_probs))
        first_indices = torch.argsort(first_probs, descending=True)[:top_first_k]
        
        # Step 2: For each first move candidate, get second move probabilities
        for first_idx in first_indices:
            first_token = idx_to_move.get(first_idx.item(), None)
            if not first_token or not first_token.startswith('m_'):
                continue
            
            first_prob = first_probs[first_idx].item()
            
            # Extend context with first move
            extended_context = context_indices + [first_idx.item()]
            extended_tensor = torch.tensor([extended_context], dtype=torch.long).to(device)
            
            # Get second move probabilities
            output2, _ = model(extended_tensor)
            second_logits = output2[0, -1]
            second_probs = torch.softmax(second_logits, dim=-1)
            
            # Get top candidates for second move
            top_second_k = min(10, len(second_probs))
            second_indices = torch.argsort(second_probs, descending=True)[:top_second_k]
            
            # Step 3: Compute joint probabilities
            for second_idx in second_indices:
                second_token = idx_to_move.get(second_idx.item(), None)
                if not second_token or not second_token.startswith('m_'):
                    continue
                
                second_prob = second_probs[second_idx].item()
                
                # Joint probability: P(m1, m2) = P(m1) × P(m2 | m1)
                joint_prob = first_prob * second_prob
                
                pairs.append((joint_prob, [first_token, second_token]))
    
    # Step 4: Sort by joint probability and return top-k
    pairs.sort(key=lambda x: x[0], reverse=True)
    return pairs[:k]
```

### For Doubles (d66, d55, etc.) - 4 Moves Needed

**Approach**: Use beam search or direct computation for quadruples

```python
def get_top_k_quadruples(model, dice_tokens, idx_to_move, move_to_idx, device, k=10):
    """
    Get top-k move quadruples with joint probabilities for doubles.
    
    Uses beam search to efficiently explore 4-move sequences.
    
    Args:
        model: Trained transformer model
        dice_tokens: List of dice tokens like ["d6", "d6"]
        idx_to_move: Token index to string mapping
        move_to_idx: Token string to index mapping
        device: PyTorch device
        k: Number of top quadruples to return
    
    Returns:
        List of (joint_prob, [move1, move2, move3, move4]) tuples, sorted by probability
    """
    model.eval()
    
    # Convert dice tokens to indices
    context_indices = [move_to_idx[token] for token in dice_tokens if token in move_to_idx]
    context_tensor = torch.tensor([context_indices], dtype=torch.long).to(device)
    
    # Beam search: keep top-k sequences at each step
    beam_width = k * 5  # Keep more candidates during search
    sequences = [(context_indices.copy(), 1.0, [])]  # (context, prob, moves)
    
    with torch.no_grad():
        for step in range(4):  # 4 moves for doubles
            candidates = []
            
            for context, prob, moves in sequences:
                context_tensor = torch.tensor([context], dtype=torch.long).to(device)
                output, _ = model(context_tensor)
                logits = output[0, -1]
                probs = torch.softmax(logits, dim=-1)
                
                # Get top candidates for this step
                top_k_step = min(beam_width, len(probs))
                top_indices = torch.argsort(probs, descending=True)[:top_k_step]
                
                for idx in top_indices:
                    token = idx_to_move.get(idx.item(), None)
                    if not token or not token.startswith('m_'):
                        continue
                    
                    step_prob = probs[idx].item()
                    joint_prob = prob * step_prob
                    new_context = context + [idx.item()]
                    new_moves = moves + [token]
                    
                    candidates.append((new_context, joint_prob, new_moves))
            
            # Keep top beam_width sequences
            candidates.sort(key=lambda x: x[1], reverse=True)
            sequences = candidates[:beam_width]
    
    # Format results
    # Only return sequences that naturally finished (hit <EOM> or <NOMOVE>) or reached max length
    quadruples = [(prob, moves) for _, prob, moves in sequences if len(moves) == 4 or (moves and moves[-1] == '<NOMOVE>')]
    quadruples.sort(key=lambda x: x[0], reverse=True)
    return quadruples[:k]
```

### Smart Sequence Termination

The beam search logic handles special tokens to support variable-length turns:

1.  **`<EOM>` Handling**:
    *   If the model predicts `<EOM>`, it signals the end of a turn (e.g., only 2 moves possible out of 4).
    *   The search path stops extending but is preserved as a valid "finished" candidate.
    *   This allows the model to correctly predict partial turns.

2.  **`<NOMOVE>` Handling**:
    *   If the model predicts `<NOMOVE>` as the first token, it signals no legal moves are available.
    *   This path is preserved as a valid candidate.

3.  **Incomplete Sequences**:
    *   Sequences that don't end in `<EOM>`/`<NOMOVE>` and haven't reached the maximum move count (2 or 4) are generally discarded or ranked lower, as they represent incomplete thoughts.

### Joint Probability Calculation

**From autoregressive model**:
```
P(m_ab, m_cd | d6, d1) = P(m_ab | d6, d1) × P(m_cd | d6, d1, m_ab)
```

**For quadruples**:
```
P(m_ab, m_cd, m_ef, m_gh | d6, d6) = 
    P(m_ab | d6, d6) × 
    P(m_cd | d6, d6, m_ab) × 
    P(m_ef | d6, d6, m_ab, m_cd) × 
    P(m_gh | d6, d6, m_ab, m_cd, m_ef)
```

### Updated `predict_backgammon_moves()` Function

```python
def predict_backgammon_moves(model, game_history, dice_roll, idx_to_move, move_to_idx, device, top_k=10):
    """
    Generate top-k move sequence predictions (pairs or quadruples).
    
    Args:
        model: Trained transformer model
        game_history: Complete game token sequence
        dice_roll: Current dice roll as string "66" or "41"
        idx_to_move: Token index to string mapping
        move_to_idx: Token string to index mapping
        device: PyTorch device
        top_k: Number of top sequences to return
    
    Returns:
        List of (joint_prob, move_sequence) tuples, sorted by probability
        - For non-doubles: [(prob, [m_ab, m_cd]), ...]
        - For doubles: [(prob, [m_ab, m_cd, m_ef, m_gh]), ...]
    """
    # Split dice into atomic tokens
    dice_tokens = split_dice(dice_roll)  # "66" -> ["d6", "d6"]
    
    # Build full context: game history + dice tokens
    full_context = game_history + dice_tokens
    
    # Determine if doubles
    is_double = is_doubles(dice_roll)
    
    # Get predictions based on dice type
    if is_double:
        # Doubles: need 4 moves
        sequences = get_top_k_quadruples(
            model, dice_tokens, idx_to_move, move_to_idx, device, k=top_k
        )
    else:
        # Non-doubles: need 2 moves
        sequences = get_top_k_pairs(
            model, dice_tokens, idx_to_move, move_to_idx, device, k=top_k
        )
    
    return sequences
```

### Helper Functions Needed

```python
def split_dice(dice_roll):
    """Split dice roll string into atomic dice tokens."""
    if len(dice_roll) == 2:
        return [f"d{dice_roll[0]}", f"d{dice_roll[1]}"]
    else:
        return [f"d{dice_roll}"]

def is_doubles(dice_roll):
    """Check if dice roll is doubles."""
    return len(dice_roll) == 2 and dice_roll[0] == dice_roll[1]
```

---

## Game Engine Integration

### Sequence Validation

The game engine validates entire sequences, not individual moves:

```python
def get_move_sequence(model, board_state, dice_roll, idx_to_move, move_to_idx, device):
    """
    Get legal move sequence from model predictions.
    
    Args:
        model: Trained transformer model
        board_state: Current board state
        dice_roll: Dice roll string "66" or "41"
        idx_to_move: Token index to string mapping
        move_to_idx: Token string to index mapping
        device: PyTorch device
    
    Returns:
        List of move tokens if legal sequence found, None otherwise
    """
    # Get top-k sequences from model
    top_sequences = predict_backgammon_moves(
        model, board_state.game_history, dice_roll,
        idx_to_move, move_to_idx, device, top_k=10
    )
    
    # Validate each sequence in order
    for joint_prob, move_sequence in top_sequences:
        if is_legal_sequence(board_state, move_sequence, dice_roll):
            return move_sequence  # First legal sequence wins
    
    # Fallback: no legal sequences found
    return None

def is_legal_sequence(board_state, move_sequence, dice_roll):
    """
    Validate entire move sequence is legal.
    
    Args:
        board_state: Current board state
        move_sequence: List of move tokens like ["m_ab", "m_cd"]
        dice_roll: Dice roll string
    
    Returns:
        True if entire sequence is legal, False otherwise
    """
    # Create temporary board copy
    temp_board = board_state.copy()
    
    # Apply each move in sequence
    for move_token in move_sequence:
        # Extract move from token (m_ab -> from 'a' to 'b')
        from_point, to_point = parse_move_token(move_token)
        
        # Check if move is legal
        if not temp_board.is_legal_move(from_point, to_point, dice_roll):
            return False
        
        # Apply move to temporary board
        temp_board.apply_move(from_point, to_point)
    
    return True
```

### Integration Points

1. **Model prediction**: Returns sequences, not single moves
2. **Sequence validation**: Validates entire sequences
3. **Fallback handling**: If no legal sequences, fall back to search AI
4. **Move application**: Apply entire sequence to board

---

## Edge Cases

### 1. Partial Moves (Can't Use All Dice)

**Scenario**: Player rolls `d66` but can only make 2 moves (not 4).

**Handling**:
- Model predicts 4 moves: `[m_ab, m_cd, m_ef, m_gh]`
- Game engine validates: Only `[m_ab, m_cd]` are legal
- **Solution**: Accept partial sequence, stop when no more legal moves

**Implementation**:
```python
def get_partial_legal_sequence(board_state, move_sequence, dice_roll):
    """Get longest legal prefix of move sequence."""
    legal_moves = []
    temp_board = board_state.copy()
    
    for move_token in move_sequence:
        from_point, to_point = parse_move_token(move_token)
        if temp_board.is_legal_move(from_point, to_point, dice_roll):
            legal_moves.append(move_token)
            temp_board.apply_move(from_point, to_point)
        else:
            break  # Stop at first illegal move
    
    return legal_moves if legal_moves else None
```

### 2. No Moves Available (Blocked)

**Scenario**: Player rolls `d66` but all pieces are blocked.

**Handling**:
- Model should predict `<NOMOVE>` `<EOM>`
- Game engine validates: No legal moves exist
- **Solution**: Accept `<NOMOVE>`, skip turn

**Training data**: Should include examples of `<NOMOVE>` for blocked positions.

### 3. Odd-Length Move Strings

**Scenario**: Move string has odd number of characters (shouldn't happen in valid games).

**Handling**:
- `split_moves()` handles gracefully
- Last character treated as single move: `m_x`
- Log warning for debugging

**Implementation**:
```python
def split_moves(move_str):
    tokens = []
    for i in range(0, len(move_str), 2):
        if i + 1 < len(move_str):
            pair = move_str[i:i+2]
            tokens.append(f"m_{pair}")
        else:
            # Odd length - log warning
            print(f"Warning: Odd-length move string: {move_str}")
            tokens.append(f"m_{move_str[i]}")
    return tokens
```

### 4. Invalid Sequences

**Scenario**: Model predicts sequence that's illegal.

**Handling**:
- Game engine validates entire sequence
- If illegal, try next sequence from top-k
- If all sequences illegal, fall back to search AI

**Implementation**: Already handled in `get_move_sequence()` function.

### 5. Missing `<EOM>` Tokens

**Scenario**: Old training data doesn't have `<EOM>` tokens.

**Handling**:
- Converter should always add `<EOM>` after moves
- Training data should be regenerated with new converter
- Model learns to predict `<EOM>` naturally

### 6. Dice Roll Format Variations

**Scenario**: SGF files might have different dice formats.

**Handling**:
- Parser should handle: `"66"`, `"6-6"`, `"6,6"`, etc.
- Normalize to two-digit string before splitting
- Log warnings for unexpected formats

---

## Testing Strategy

### 1. Converter Testing

**Test cases**:

1. **Dice splitting**:
   - `"66"` → `["d6", "d6"]` ✓
   - `"41"` → `["d4", "d1"]` ✓
   - `"33"` → `["d3", "d3"]` ✓
   - `"12"` → `["d1", "d2"]` ✓

2. **Move splitting**:
   - `"abcd"` → `["m_ab", "m_cd"]` ✓
   - `"abcdefgh"` → `["m_ab", "m_cd", "m_ef", "m_gh"]` ✓
   - `"xz"` → `["m_xz"]` ✓
   - `"a"` → `["m_a"]` (odd length) ✓

3. **Special tokens**:
   - After moves: `<EOM>` added ✓
   - No moves: `<NOMOVE>` `<EOM>` added ✓
   - Game boundaries: `<STARTGAME>` and `<EOFG>` preserved ✓

4. **Complete turns**:
   - `"66abcdefgh"` → `["d6", "d6", "m_ab", "m_cd", "m_ef", "m_gh", "<EOM>"]` ✓
   - `"41xyzw"` → `["d4", "d1", "m_xy", "m_zw", "<EOM>"]` ✓
   - `"66"` → `["d6", "d6", "<NOMOVE>", "<EOM>"]` ✓

**Test files**: Create sample SGF files with various scenarios.

### 2. Training Testing

**Test cases**:

1. **Vocabulary creation**:
   - Verify atomic dice tokens (`d1`-`d6`) are created ✓
   - Verify atomic move tokens (`m_xx`) are created ✓
   - Verify special tokens (`<EOM>`, `<NOMOVE>`) are included ✓
   - Check vocabulary size is smaller than before ✓

2. **Dataset loading**:
   - Verify tokens are correctly indexed ✓
   - Verify sequences are created correctly ✓
   - Check for invalid token indices ✓

3. **Training loop**:
   - Monitor loss decreases ✓
   - Verify model learns to predict `<EOM>` ✓
   - Check model can predict move pairs ✓

**Test data**: Use small sample dataset first, then full dataset.

### 3. Inference Testing

**Test cases**:

1. **Pair prediction** (non-doubles):
   - Input: `d6 d1`
   - Output: Top-k pairs like `[(0.15, [m_ab, m_cd]), (0.12, [m_ab, m_ef]), ...]` ✓
   - Verify joint probabilities are computed correctly ✓

2. **Quadruple prediction** (doubles):
   - Input: `d6 d6`
   - Output: Top-k quadruples like `[(0.08, [m_ab, m_cd, m_ef, m_gh]), ...]` ✓
   - Verify beam search works correctly ✓

3. **Sequence validation**:
   - Test with legal sequences ✓
   - Test with illegal sequences ✓
   - Test fallback to next sequence ✓

**Test scenarios**: Create test board positions with known legal moves.

### 4. Integration Testing

**Test cases**:

1. **End-to-end flow**:
   - SGF → Converter → Training → Inference → Game engine ✓
   - Verify moves are legal ✓
   - Verify game plays correctly ✓

2. **Edge cases**:
   - Blocked positions (`<NOMOVE>`) ✓
   - Partial moves ✓
   - Invalid sequences ✓

**Test games**: Play complete games and verify correctness.

---

## Benefits and Rationale

### 1. Smaller Vocabulary

**Before**:
- Dice tokens: 21 (`d11`-`d66`)
- Move tokens: Thousands (every unique sequence)

**After**:
- Dice tokens: 6 (`d1`-`d6`)
- Move tokens: ~676 pairs (26×26, fewer valid)

**Benefit**: Model can learn all atomic patterns, composes unseen sequences.

### 2. Better Generalization

**Before**: Model memorizes `d66` → `m_abcd` as single pattern.

**After**: Model learns:
- `d6` → `m_ab` patterns
- `m_ab` + `m_cd` composition
- Can predict `m_ab` `m_cd` `m_ef` `m_gh` even if never seen together

**Benefit**: Handles billions of possible move combinations.

### 3. Natural Structure

**Before**: One token = entire dice roll or move sequence.

**After**: Each token = one die or one move.

**Benefit**: Matches game mechanics exactly (each die enables one move).

### 4. Joint Probability Learning

**Before**: Predicts single moves independently.

**After**: Predicts coherent sequences with joint probabilities.

**Benefit**: Model learns strategic move combinations.

### 5. Training Efficiency

**Before**: Needs to see every unique sequence.

**After**: Learns atomic patterns, composes sequences.

**Benefit**: Faster convergence, better performance with less data.

---

## Implementation Checklist

### Converter (`Backgammon_SGF_to_TXT_Converter.py`)

- [ ] Create `split_dice()` function
  - [ ] Handle two-digit dice strings
  - [ ] Return list of atomic dice tokens
  - [ ] Handle edge cases (single digit, invalid format)

- [ ] Create `split_moves()` function
  - [ ] Split move strings into pairs
  - [ ] Handle odd-length strings
  - [ ] Return list of atomic move tokens

- [ ] Create `is_doubles()` helper function
  - [ ] Check if dice roll is doubles
  - [ ] Return boolean

- [ ] Modify `parse_sgf_moves()` function
  - [ ] Extract dice and split into atomic tokens
  - [ ] Extract moves and split into pairs
  - [ ] Add `<EOM>` after moves
  - [ ] Add `<NOMOVE>` `<EOM>` when no moves
  - [ ] Handle all SGF format variations

- [ ] Update vocabulary creation
  - [ ] Verify atomic tokens are included
  - [ ] Verify `<EOM>` and `<NOMOVE>` are included
  - [ ] Check vocabulary size reduction

- [ ] Test with sample SGF files
  - [ ] Test dice splitting (all combinations)
  - [ ] Test move splitting (2, 4, 6, 8 characters)
  - [ ] Test special token insertion
  - [ ] Test edge cases

### Training (`BackgammonBrain_Parallel_11_5_25.py`)

- [ ] Verify `create_move_to_idx_from_text()` works with new format
  - [ ] Check atomic tokens are extracted correctly
  - [ ] Verify special tokens are included
  - [ ] Check vocabulary statistics

- [ ] Verify `BackgammonMovesDataset` handles new format
  - [ ] Check tokenization works correctly
  - [ ] Verify sequences are created properly
  - [ ] Check for invalid token indices

- [ ] No changes needed to training loop
  - [ ] Autoregressive training works automatically
  - [ ] Loss calculation unchanged
  - [ ] Backpropagation unchanged

- [ ] Test training with new tokenized data
  - [ ] Monitor loss decreases
  - [ ] Verify model learns patterns
  - [ ] Check model can predict `<EOM>`

### Inference

- [ ] Create `split_dice()` helper function
  - [ ] Split dice roll string into tokens
  - [ ] Handle all dice combinations

- [ ] Create `is_doubles()` helper function
  - [ ] Check if dice roll is doubles
  - [ ] Used for determining sequence length

- [ ] Create `get_top_k_pairs()` function
  - [ ] Get top-k first moves
  - [ ] For each, get top-k second moves
  - [ ] Compute joint probabilities
  - [ ] Return top-k pairs sorted by probability

- [ ] Create `get_top_k_quadruples()` function
  - [ ] Use beam search for 4-move sequences
  - [ ] Compute joint probabilities
  - [ ] Return top-k quadruples sorted by probability

- [ ] Modify `predict_backgammon_moves()` function
  - [ ] Detect doubles vs non-doubles
  - [ ] Call appropriate prediction function
  - [ ] Return sequences instead of single tokens
  - [ ] Handle edge cases

- [ ] Update `BackgammonMovePredictor` class
  - [ ] Modify `predict_moves()` to return sequences
  - [ ] Update `get_legal_moves()` to validate sequences
  - [ ] Handle partial sequences

### Game Engine Integration

- [ ] Create `is_legal_sequence()` function
  - [ ] Validate entire move sequence
  - [ ] Check each move in sequence
  - [ ] Return boolean

- [ ] Create `get_partial_legal_sequence()` function
  - [ ] Get longest legal prefix
  - [ ] Handle partial moves gracefully

- [ ] Update game engine move selection
  - [ ] Get sequences from model
  - [ ] Validate sequences
  - [ ] Use first legal sequence
  - [ ] Fallback to search AI if needed

### Testing

- [ ] Converter testing
  - [ ] Test dice splitting (all combinations)
  - [ ] Test move splitting (various lengths)
  - [ ] Test special token insertion
  - [ ] Test edge cases

- [ ] Training testing
  - [ ] Verify vocabulary creation
  - [ ] Check dataset loading
  - [ ] Monitor training progress

- [ ] Inference testing
  - [ ] Test pair prediction (non-doubles)
  - [ ] Test quadruple prediction (doubles)
  - [ ] Verify joint probabilities
  - [ ] Test sequence validation

- [ ] Integration testing
  - [ ] End-to-end flow
  - [ ] Edge cases
  - [ ] Complete games

---

## Summary

This document provides a complete guide for implementing atomic tokenization in your backgammon AI system. The key changes are:

1. **Converter**: Split dice and moves into atomic tokens, add `<EOM>` and `<NOMOVE>`
2. **Training**: No changes needed - autoregressive training works automatically
3. **Inference**: Predict pairs/quadruples with joint probabilities instead of single moves
4. **Game Engine**: Validate entire sequences, handle partial moves gracefully

The implementation reduces vocabulary size dramatically while enabling better generalization to unseen move combinations. The model learns atomic patterns and composes them into coherent strategic sequences.

