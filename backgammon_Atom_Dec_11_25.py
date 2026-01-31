"""
JMR's Backgammon Game with LLM Integration

This is a Pygame-based backgammon implementation featuring:
- Human vs AI gameplay
- LLM-powered AI (BackgammonMovePredictor)
- Traditional search-based AI fallback
- Complete game state tracking for LLM context
- Debug-friendly architecture with comprehensive logging

===============================================================================
🎯 QUICK DEBUGGING GUIDE - READ THIS FIRST WHEN ISSUES OCCUR!
===============================================================================

1. COMMON FAILURE POINTS:
   - LLM token parsing (token_to_positions) - #1 cause of LLM failures
   - Board state corruption - check board[:] copying
   - Dice validation - ensure dice match available moves
   - Move legality - is_valid_move() function

2. DEBUG WORKFLOW:
   a) Check terminal output for emojis (✅❌🔄⚠️) indicating success/failure
   b) Look for "LLM_AI thinking" vs "Search AI evaluating"
   c) Watch game_history_tokens for proper token format
   d) Verify board state with print(board) if needed

3. LLM-SPECIFIC DEBUGGING:
   - "Could not parse LLM move" = token format issue
   - "LLM move not legal" = board state mismatch
   - "No legal LLM moves found" = LLM giving invalid suggestions
   - Falls back to search AI automatically

4. SEARCH AI DEBUGGING:
   - "Search AI evaluating moves" = AI is working
   - "Move X→Y scores: Z.ZZ" = evaluation working
   - "Search AI chose move" = successful move selection
   - "Search AI found no moves" = board state issue

5. KEY VARIABLES TO WATCH:
   - player_turn: 1=human, -1=AI
   - dice_rolled: True when dice available
   - game_history_tokens: Complete move history
   - board: Current piece positions (-=black, +=white)
   - bar: [white_on_bar, black_on_bar]

6. WHEN THINGS BREAK:
   - LLM failures → Check token_to_positions()
   - Search AI crashes → Check board copying (board[:])
   - Invalid moves → Check is_valid_move() logic
   - No moves available → Check dice and board state

DEBUG NOTES:
- Game state is tracked in game_history_tokens for LLM context
- All moves are logged with format: d{dice}m{move_token}
- LLM failures fall back to search AI automatically
- Check terminal output for detailed move reasoning
"""

import pygame
import random
import time
import copy
import sys
import os

# === LLM INTEGRATION SETUP ===
# Use standalone predictor for inference - handles device mapping automatically
# This allows the game to use AI-powered move prediction without complex GPU setup
try:
    from BackgammonMovePredictor_Standalone_Atom import BackgammonMovePredictor
    LLM_AVAILABLE = True
    print("✅ Standalone LLM predictor available for enhanced gameplay (Atomic Tokenization)")
    print("   (Handles device mapping automatically - works on any hardware)")
except ImportError:
    print("⚠️  Warning: Could not import BackgammonMovePredictor_Standalone_Atom.")
    print("   Make sure BackgammonMovePredictor_Standalone_Atom.py is in the same directory.")
    LLM_AVAILABLE = False
    BackgammonMovePredictor = None

# === GAME CONSTANTS ===
# Display and board dimensions
WIDTH, HEIGHT = 800, 600  # Main game area
BOARD_SIZE = 24  # Backgammon board has 24 playable positions (0-23)
PIECE_SIZE = 30  # Size of game pieces in pixels

# Letters mapping for display (SGF-style a-x)
LETTERS = [chr(ord('a') + i) for i in range(BOARD_SIZE)]

def idx_to_letter(pos):
    # Convert internal position to standard SGF-style letter label
    # Standard backgammon SGF notation used in game databases:
    # a-x = positions 1-24 on the board (a=1 bottom right, x=24 top right)
    # y = bar (where pieces go when hit, shared by both players)
    # z = bearing off (moving pieces off the board)
    # Both players use 'y' for bar and 'z' for bearing off
    if pos == -1 or pos == 25:  # Both bars use 'y'
        return 'y'
    if pos == 0 or pos == 26:   # Both bearing off use 'z'
        return 'z'
    if 1 <= pos <= BOARD_SIZE:
        # Position equals label: pos 1 = 'a', pos 24 = 'x'
        return LETTERS[pos - 1]
    return str(pos)

def letter_to_idx(s, player=None):
    # Convert standard SGF-style letter label to internal position
    # 'y' = bar: returns white bar (25) for white player, black bar (-1) for black player
    # 'z' = bearing off: returns white bearing off (0) for white, black bearing off (25) for black
    # a-x = positions 1-24 on board
    if s == 'y':  # bar position depends on player
        return 25 if player == 1 else -1
    if s == 'z':  # bearing off position depends on player
        return 0 if player == 1 else 25
    if len(s) == 1 and 'a' <= s <= 'x':
        # Label equals position: 'a' -> pos 1, 'x' -> pos 24
        return ord(s) - ord('a') + 1
    # Fallback: try numeric position directly
    try:
        return int(s)
    except Exception:
        return None

# Colors for UI and pieces
WHITE_ISH = (220, 220, 220)  # Background
LIGHT_GRAY = (200, 200, 200) # Board spaces
WHITE = (255, 255, 255)      # White pieces
BLACK = (0, 0, 0)            # Black pieces, text
LIGHT_BLUE = (173, 216, 230) # White pieces (light blue)
PINK = (255, 192, 203)       # Black pieces (pink)
RED = (255, 0, 0)            # Error/highlights
BLUE = (0, 0, 255)           # AI indicators
GREEN = (0, 255, 0)          # Success indicators

# Home board boundaries (where bearing off happens)
# Position = Label. Blue bears off from 1-6, Red bears off from 19-24
# Blue moves counter-clockwise (24→1), Red moves clockwise (1→24)
BLUE_HOME_START = 1   # Blue bears off from positions 1-6 (labels 1-6)
BLUE_HOME_END = 6
RED_HOME_START = 19  # Red bears off from positions 19-24 (labels 19-24)
RED_HOME_END = 24

# === PYGAME INITIALIZATION ===
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT+200))  # Extra space for UI
pygame.display.set_caption("JMR's Backgammon")

# === GAME STATE REPRESENTATION ===
# board: List of 25 integers representing piece counts at each position (index 0 unused, 1-24 are positions)
# Negative = black pieces, Positive = white pieces, Zero = empty
# Position 1 = Label 1 (a) = bottom right, Position 24 = Label 24 (x) = top right
# Internal position numbers match visible labels for easy debugging
board = [0] * (BOARD_SIZE + 1)  # Index 0 unused, positions 1-24

# bar: Pieces waiting to re-enter the game
# bar[0] = white pieces on bar, bar[1] = black pieces on bar
bar = [0, 0]

# === INITIAL BOARD SETUP ===
# Standard backgammon starting positions (position = label)
# Blue pieces: start at 24, move counter-clockwise (24→1), bear off to 0
# Red pieces: start at 1, move clockwise (1→24), bear off to 25
board[24] = 2    # Blue: 2 pieces at position 24 (label 24, 'x')
board[13] = 5    # Blue: 5 pieces at position 13 (label 13, 'm')
board[8] = 3     # Blue: 3 pieces at position 8 (label 8, 'h')
board[6] = 5     # Blue: 5 pieces at position 6 (label 6, 'f')

board[1] = -2    # Red: 2 pieces at position 1 (label 1, 'a')
board[12] = -5   # Red: 5 pieces at position 12 (label 12, 'l')
board[17] = -3   # Red: 3 pieces at position 17 (label 17, 'q')
board[19] = -5   # Red: 5 pieces at position 19 (label 19, 's')

white_pieces_off = 0
black_pieces_off = 0

# Track LLM failures for game statistics
llm_failures = 0

def display_initial_rolls(player_die, ai_die):
    """Display the initial die rolls to determine who goes first"""
    # Clear the display area
    roll_rect = pygame.Rect(0, HEIGHT + 50, WIDTH, 100)
    pygame.draw.rect(screen, WHITE_ISH, roll_rect)

    font = pygame.font.Font(None, 36)
    font_small = pygame.font.Font(None, 24)

    # Display player die on the left
    player_text = font.render(f"Player: {player_die}", True, BLACK)
    screen.blit(player_text, (50, HEIGHT + 60))

    # Display AI die on the right
    ai_text = font.render(f"AI: {ai_die}", True, BLACK)
    screen.blit(ai_text, (WIDTH - 150, HEIGHT + 60))

    # Display result or instruction
    if player_die == ai_die:
        tie_text = font_small.render("Tie! Press SPACEBAR to roll again", True, (255, 0, 0))
        screen.blit(tie_text, (WIDTH//2 - 150, HEIGHT + 100))
    elif player_die > ai_die:
        result_text = font_small.render("Player goes first!", True, (0, 150, 0))
        screen.blit(result_text, (WIDTH//2 - 100, HEIGHT + 100))
    else:
        result_text = font_small.render("AI goes first!", True, (0, 150, 0))
        screen.blit(result_text, (WIDTH//2 - 80, HEIGHT + 100))

    pygame.display.flip()

def display_roll_prompt():
    """Display prompt to press space to roll initial dice"""
    # Clear the display area
    roll_rect = pygame.Rect(0, HEIGHT + 50, WIDTH, 100)
    pygame.draw.rect(screen, WHITE_ISH, roll_rect)

    font = pygame.font.Font(None, 32)
    font_small = pygame.font.Font(None, 24)

    # Display prompt
    prompt_text = font.render("Press SPACEBAR to roll dice and determine who goes first", True, (0, 100, 200))
    screen.blit(prompt_text, (WIDTH//2 - 300, HEIGHT + 70))

    pygame.display.flip()

# Function to draw the board
def draw_board(board, bar, highlight_positions=None):
    #want to just cover the top HEIGHT of the screen
    # Add extra clearance for highlighted pieces that extend beyond normal boundaries
    extra_clearance = 10 if highlight_positions else 0
    rect = pygame.Rect(0, 0, WIDTH, HEIGHT + extra_clearance)
    pygame.draw.rect(screen, WHITE_ISH, rect)   

    for i in range(1, BOARD_SIZE + 1):
        # Bottom row: pos 1-12, RIGHT to LEFT (pos 1 at right)
        # Top row: pos 13-24, left to right (pos 24 at right)
        if i <= 12:
            x = (12 - i) * (WIDTH // 12)
            y = HEIGHT // 2
            opposite_color = BLUE if i % 2 == 0 else RED
            # Draw triangles pointing up from the bottom
            pygame.draw.polygon(screen, opposite_color, [(x, y + HEIGHT // 2), (x + WIDTH // 12, y + HEIGHT // 2), (x + WIDTH // 24, y)])
            # Label for bottom row
            lbl_pos = (x + WIDTH // 24, y + HEIGHT // 2 - 18)
        else:
            x = (i - 13) * (WIDTH // 12)
            y = 0
            color = BLUE if i % 2 == 0 else RED
            # Draw triangles pointing down from the top
            pygame.draw.polygon(screen, color, [(x, y), (x + WIDTH // 12, y), (x + WIDTH // 24, y + HEIGHT // 2)])
            # Label for top row
            lbl_pos = (x + WIDTH // 24, y + 4)

        # Label: position = label
        lbl_font = pygame.font.Font(None, 18)
        label = f"{idx_to_letter(i)}{i}"
        text_surf = lbl_font.render(label, True, BLACK)
        screen.blit(text_surf, lbl_pos)

    for i in range(1, BOARD_SIZE + 1):
        # Bottom row: pos 1-12, RIGHT to LEFT (pos 1 at right)
        # Top row: pos 13-24, left to right (pos 24 at right)
        if i <= 12:
            x = (12 - i) * (WIDTH // 12) + (WIDTH // 24)
            y = HEIGHT - PIECE_SIZE // 2
        else:
            x = (i - 13) * (WIDTH // 12) + (WIDTH // 24)
            y = PIECE_SIZE // 2

        color = LIGHT_BLUE if board[i] > 0 else PINK
        for j in range(abs(board[i])):
            radius = PIECE_SIZE // 2
            if highlight_positions and i in highlight_positions:
                # Highlight the piece CLOSEST TO CENTER of the board
                # For both rows: the last piece drawn (highest in the stack)
                should_highlight = (j == abs(board[i]) - 1)
                if should_highlight:
                    radius = PIECE_SIZE // 2 + 5  # Make highlighted piece bigger
            if i <= 12:
                pygame.draw.circle(screen, color, (x, y - j * PIECE_SIZE), radius)
            else:
                pygame.draw.circle(screen, color, (x, y + j * PIECE_SIZE), radius)

    # Draw the bar (vertical line in center)
    bar_center_x = WIDTH // 2
    bar_y = HEIGHT // 2

    # Draw thicker black vertical bar line
    pygame.draw.line(screen, BLACK, (bar_center_x, 0), (bar_center_x, HEIGHT), 8)

    # Draw white pieces on bar (left side of bar)
    for i in range(bar[0]):
        radius = PIECE_SIZE // 2
        if highlight_positions and 25 in highlight_positions:
            # Highlight the RIGHTMOST piece (closest to center, i == bar[0]-1 for white bar)
            if i == bar[0] - 1:
                radius = PIECE_SIZE // 2 + 5  # Highlight rightmost white bar piece
        x_pos = bar_center_x - PIECE_SIZE - (i * PIECE_SIZE)  # Pieces to the left, spaced horizontally
        pygame.draw.circle(screen, LIGHT_BLUE, (x_pos, bar_y), radius)

    # Draw black pieces on bar (right side of bar)
    for i in range(bar[1]):
        radius = PIECE_SIZE // 2
        if highlight_positions and -1 in highlight_positions:
            # Highlight the LEFTMOST piece (closest to center, i == 0 for black bar)
            if i == 0:
                radius = PIECE_SIZE // 2 + 5  # Highlight leftmost black bar piece
        x_pos = bar_center_x + PIECE_SIZE + (i * PIECE_SIZE)  # Pieces to the right, spaced horizontally
        pygame.draw.circle(screen, PINK, (x_pos, bar_y), radius)


def draw_dice(dice1, dice2):
    dice_size = 50
    dice_padding = 10
    dice_x = WIDTH // 2 - dice_size - dice_padding
    dice_y = HEIGHT // 2 - dice_size // 2

    # Clear larger area to ensure old dots are fully covered (covers full dot extent)
    # Dots extend from x-5 to x+55, so clear from x-5 to x+55 (width 60)
    pygame.draw.rect(screen, WHITE, (dice_x - 5, dice_y - 5, dice_size + 10, dice_size + 10))
    pygame.draw.rect(screen, BLACK, (dice_x, dice_y, dice_size, dice_size), 2)
    if dice1 > 0 and dice1 < 7:
        draw_dice_dots(dice1, dice_x, dice_y, dice_size)

    dice_x = WIDTH // 2 + dice_padding
    # Clear larger area to ensure old dots are fully covered (covers full dot extent)
    pygame.draw.rect(screen, WHITE, (dice_x - 5, dice_y - 5, dice_size + 10, dice_size + 10))
    pygame.draw.rect(screen, BLACK, (dice_x, dice_y, dice_size, dice_size), 2)
    if dice2 > 0 and dice2 < 7:
        draw_dice_dots(dice2, dice_x, dice_y, dice_size)

    # Redraw bar pieces so dice clear rect doesn't hide them
    bar_center_x = WIDTH // 2
    bar_y = HEIGHT // 2
    for i in range(bar[0]):  # White bar (left)
        radius = PIECE_SIZE // 2
        x_pos = bar_center_x - PIECE_SIZE - (i * PIECE_SIZE)
        pygame.draw.circle(screen, LIGHT_BLUE, (x_pos, bar_y), radius)
    for i in range(bar[1]):  # Black bar (right)
        radius = PIECE_SIZE // 2
        x_pos = bar_center_x + PIECE_SIZE + (i * PIECE_SIZE)
        pygame.draw.circle(screen, PINK, (x_pos, bar_y), radius)

# Function to draw the dots on the dice
def draw_dice_dots(value, x, y, size):
    dot_size = size // 5
    padding = size // 10

    positions = {
        1: [(size // 2, size // 2)],
        2: [(padding, padding), (size - padding, size - padding)],
        3: [(padding, padding), (size // 2, size // 2), (size - padding, size - padding)],
        4: [(padding, padding), (padding, size - padding), (size - padding, padding), (size - padding, size - padding)],
        5: [(padding, padding), (padding, size - padding), (size // 2, size // 2), (size - padding, padding), (size - padding, size - padding)],
        6: [(padding, padding), (padding, size // 2), (padding, size - padding), (size - padding, padding), (size - padding, size // 2), (size - padding, size - padding)]
    }
    for pos in positions[value]:
        pygame.draw.circle(screen, BLACK, (x + pos[0], y + pos[1]), dot_size)


def roll_dice():
    return random.randint(1, 6), random.randint(1, 6)


def get_llm_move_context(dice1, dice2, game_history_tokens=None):
    """
    Prepare game context for LLM move prediction.

    Combines the full game history with current dice roll to provide
    complete context for strategic move prediction, similar to chess.

    Args:
        dice1, dice2: Current dice values
        game_history_tokens: List of previous game tokens

    Returns:
        Tuple (game_history, current_dice_token)
    """
    # CRITICAL: Dice must be in canonical order (larger die first)
    # Training data uses d53 not d35, d64 not d46, etc.
    if dice1 >= dice2:
        dice_token = f"d{dice1}{dice2}"
    else:
        dice_token = f"d{dice2}{dice1}"

    if game_history_tokens is None:
        game_history_tokens = ["<STARTGAME>"]

    return game_history_tokens, dice_token


def perform_search_ai_with_move_capture(dice, moves_left_ai, player):
    """
    === SEARCH AI WITH MOVE CAPTURE ===
    Execute search_AI moves while capturing what moves were actually made.

    This creates a modified version of ai_move that returns the moves it made
    so they can be converted to tokens for LLM history tracking.

    DEBUG NOTES:
    - This function evaluates all possible moves and picks the best one
    - Uses minimax-like evaluation to score board positions
    - Critical: board copying must be done correctly (was the bug here)
    - Returns the actual move made for history tracking

    Args:
        dice: List of dice values available for moves
        moves_left_ai: Number of moves remaining for AI this turn
        player: Which player the AI is playing for (1=white, -1=black)

    Returns:
        (start_pos, end_pos) tuple for the move made, or None if no move
    """
    global black_pieces_off, white_pieces_off
    # === SEARCH AI ALGORITHM ===
    # We'll modify the ai_move logic to capture and return the move
    # For simplicity, let's implement a basic version that finds the best move
    # and returns it before executing

    best_move = None
    min_score = float('inf')  # Always look for minimum score (after inversion for white)

    print(f"🎯 Search AI evaluating moves with dice: {dice} (playing as {'black' if player == -1 else 'white'})")

    # === CHECK FOR MANDATORY BAR MOVES ===
    bar_index = 1 if player == -1 else 0
    bar_start = -1 if player == -1 else 25  # FIX: black uses -1, white uses 25 for bar (consistent with letter_to_idx)
    must_move_from_bar = False
    if bar[bar_index] > 0:
        for end in range(1, BOARD_SIZE + 2):  # Check positions 1-25
            if is_valid_move(bar_start, end, dice, player):
                must_move_from_bar = True
                break

    # === MOVE EVALUATION LOOP ===
    # Find the best move by evaluating all possible moves
    starts = [bar_start] + list(range(1, BOARD_SIZE + 1))
    for start in starts:
        # Skip invalid starting positions
        if start == bar_start and bar[bar_index] == 0:  # No pieces on bar for this player
            continue
        if start != bar_start and (board[start] * player) <= 0:  # Not a piece of the correct color
            continue
        # If must move from bar, skip non-bar moves
        if must_move_from_bar and start != bar_start:
            continue

        for end in range(0, BOARD_SIZE + 2):  # Include positions 0-25 (bearing off: white=0, black=25)
            if is_valid_move(start, end, dice, player):
                # === BOARD STATE BACKUP (CRITICAL) ===
                # Save current board state before trying the move
                # DEBUG: This was the bug - was trying [row[:] for row in board] on 1D list
                original_board = board[:]  # Correct: simple list copy
                original_bar = bar[:]
                original_black_pieces_off = black_pieces_off
                original_white_pieces_off = white_pieces_off
                original_dice = dice[:]  # Backup dice state

                # === TEMPORARY MOVE EXECUTION ===
                # Make the move temporarily to evaluate the resulting position
                # This simulates what the board would look like after the move
                bar_index = 1 if player == -1 else 0  # Bar index for this player
                opponent_bar_index = 0 if player == -1 else 1  # Opponent's bar

                if start == bar_start:  # Moving from bar
                    bar[bar_index] -= 1
                else:  # Moving from board position
                    board[start] -= player  # Remove piece from start (player pieces are +player or -player)

                if end == 25 or end == 0:  # Bearing off (white=0, black=25)
                    if player == -1:
                        black_pieces_off += 1
                    else:
                        white_pieces_off += 1
                elif board[end] == -player:  # Hitting opponent piece
                    board[end] = player  # Place our piece
                    bar[opponent_bar_index] += 1  # Send opponent piece to bar
                else:  # Regular move
                    board[end] += player  # Place our piece

                # === TEMPORARY DICE UPDATE ===
                # Update dice state for evaluation of remaining moves
                if end == 25 or end == 0:  # Bearing off
                    distance = 25 - start if player == -1 else start
                    # Find which die was used
                    for i in range(len(dice)):
                        if dice[i] == distance or (dice[i] > distance and dice[i] < 7):
                            dice[i] = 7  # Mark as used
                            break
                else:
                    # Handle bar moves correctly for dice update
                    if start == 25:  # White bar
                        move_distance = 25 - end
                    elif start == -1:  # Black bar
                        move_distance = end
                    else:
                        move_distance = abs(end - start)
                        
                    for i in range(len(dice)):
                        if dice[i] == move_distance and dice[i] < 7:
                            dice[i] = 7  # Mark as used
                            break

                # === POSITION EVALUATION ===
                # Evaluate how good this board position is for the AI player
                score = evaluate_board(board, bar, player)

                # Bonus for bearing off from furthest positions
                if end == 0 or end == 25:  # Bearing off move
                    if player == 1 and end == 0:  # White bearing off from lower positions (furthest from off)
                        score -= (7 - start) * 1000  # Lower start gets much better (lower) score
                    elif player == -1 and end == 25:  # Black bearing off from lower positions (furthest from off)
                        score -= (25 - start) * 1000  # Lower start gets much better (lower) score

                # === BOARD STATE RESTORATION ===
                # Restore the board to its original state for next evaluation
                board[:] = original_board
                bar[:] = original_bar
                black_pieces_off = original_black_pieces_off
                white_pieces_off = original_white_pieces_off
                dice[:] = original_dice  # Restore dice state

                # === BEST MOVE TRACKING ===
                # Always look for minimum score (after inversion for white)
                if score < min_score:
                    min_score = score
                    best_move = (start, end)

    # === EXECUTE BEST MOVE ===
    if best_move:
        move_token = move_to_token(best_move[0], best_move[1], player)
        print(f"🤖 {move_token}")
        make_move(best_move[0], best_move[1], player, moves_left_ai, min_score)
        return best_move

    print("⚠️ No moves available")
    return None


def llm_ai_move(dice, moves_left_ai, game_history_tokens, player):
    """
    === LLM AI MOVE EXECUTION ===
    Get move recommendations from LLM and execute the first legal one.
    Updated for Atomic Tokenization.
    """
    global llm_predictor, ai_type

    # === AVAILABILITY CHECK ===
    if not llm_predictor or ai_type != "LLM_AI":
        print("LLM not available, using search_AI")
        return None

    # === CONTEXT PREPARATION ===
    # Prepare full game context for LLM (like chess implementation)
    # This includes recent moves + current dice state
    game_history, dice_token = get_llm_move_context(dice[0], dice[1], game_history_tokens)

    print(f"🧠 LLM_AI Context: {dice_token}")
    print(f"📜 Full Game History ({len(game_history_tokens)} tokens):")
    print(f"   {game_history_tokens}")  # Show complete history for LLM context
    print(f"🎯 Sending to LLM: game_history + dice_token")

    try:
        # Get ALL LLM predictions (sequences of atomic moves)
        # The predictor returns 'top_k' COMPLETE sequences, not just individual moves.
        # Example: [
        #    (0.85, ['m_ab', 'm_cd']),  # Sequence 1: Move a->b, then c->d (Confidence 85%)
        #    (0.10, ['m_ef', 'm_gh']),  # Sequence 2: Move e->f, then g->h (Confidence 10%)
        # ]
        # We try Sequence 1 first. If it's valid on the board, we use it.
        # If not, we try Sequence 2, and so on.
        all_sequences = llm_predictor.predict_moves(game_history, dice_token, top_k=5)

        print(f"🤖 LLM_AI RAW RESPONSES: {len(all_sequences)} sequences")
        
        for i, (confidence, sequence) in enumerate(all_sequences, 1):
            print(f"  {i}. {sequence} (confidence: {confidence:.4f})")

        # Iterate through candidate sequences and find the first valid one
        for confidence, sequence in all_sequences:
            print(f"🔍 Checking sequence: {sequence}")
            
            # Validate sequence
            if not sequence:
                continue
                
            # Check for NOMOVE special token
            if sequence[0] == '<NOMOVE>':
                print("✅ LLM suggested NO MOVE")
                # Verify we really have no moves
                if not has_valid_moves(player, dice):
                     print("✅ Verified: No moves possible")
                     return ('<NOMOVE>', 0)
                else:
                     print("❌ Invalid <NOMOVE> suggestion - moves are possible")
                     continue
            
            # Simulate and Execute
            # We need to simulate the sequence to see if it's valid
            # If valid, we execute it
            
            import copy
            test_board = copy.deepcopy(board)
            test_bar = copy.deepcopy(bar)
            test_dice = list(dice)
            
            valid_sequence = True
            moves_to_make = []
            
            for move_token in sequence:
                 if move_token == '<EOM>':
                     break
                 if not move_token.startswith('m_'):
                     continue
                     
                 start, end = token_to_positions(move_token, test_dice, player)
                 
                 if start is None or end is None:
                     valid_sequence = False
                     print(f"   ❌ Could not parse or illegal move: {move_token}")
                     break
                 
                 # Check legality on test board
                 # token_to_positions calls is_valid_move but on the REAL board/dice?
                 # No, token_to_positions uses the 'dice' argument passed to it, 
                 # but it accesses the global 'board' by default in is_valid_move call unless we pass board.
                 
                 # Fix: We need to call is_valid_move with test_board
                 if not is_valid_move(start, end, test_dice, player, test_board, test_bar):
                     valid_sequence = False
                     print(f"   ❌ Illegal move on simulated board: {move_token}")
                     break
                     
                 moves_to_make.append((start, end, move_token))
                 
                 # Update test state
                 # Apply move to test_board
                 move_distance = abs(end - start) if end not in [0, 25, 26] else (25 - start if player == -1 else start)
                 
                 if start == 25 or start == -1:  # From bar
                    test_bar[0 if player == 1 else 1] -= 1
                 elif start >= 1 and start <= BOARD_SIZE:
                    test_board[start] -= player
                    
                 if end == 0 or end == 25 or end == 26:  # Bearing off
                    pass
                 elif end >= 1 and end <= BOARD_SIZE:
                    if test_board[end] * player < 0 and abs(test_board[end]) == 1:  # Hit opponent
                        test_bar[1 if player == 1 else 0] += 1
                        test_board[end] = player
                    else:
                        test_board[end] += player

                 # Update dice
                 dice_used = False
                 for j in range(len(test_dice)):
                    if test_dice[j] == move_distance and test_dice[j] < 7:
                        test_dice[j] = 7
                        dice_used = True
                        break
                 
                 if not dice_used:
                     # Could be bearing off with higher die
                     for j in range(len(test_dice)):
                        if test_dice[j] > move_distance and test_dice[j] < 7:
                             test_dice[j] = 7
                             dice_used = True
                             break
                 
                 if not dice_used:
                     valid_sequence = False
                     print(f"   ❌ No matching die for move: {move_token}")
                     break
            
            # === CRITICAL: CHECK FOR INCOMPLETE TURNS ===
            # If sequence is valid so far, verify if it used all possible moves
            # We don't want "half moves" (lazy AI) where it stops but could continue
            if valid_sequence:
                # Collect remaining unused dice from simulation
                remaining_dice = [d for d in test_dice if d < 7]
                
                if remaining_dice and has_valid_moves(player, remaining_dice, test_board, test_bar):
                    # If valid moves remain, the LLM sequence is incomplete!
                    # REJECT IT to force fallback to Search AI for a full turn
                    valid_sequence = False
                    print(f"   ❌ Sequence incomplete - valid moves remain with dice {remaining_dice}")
                    # Note: We set valid_sequence=False so the loop continues to the next candidate
                    # If all candidates are incomplete, the loop finishes with valid_sequence=False
                    # and the function returns None, triggering full search AI fallback.

            if valid_sequence and moves_to_make:
                print(f"✅ Sequence accepted! Executing {len(moves_to_make)} moves.")
                moves_executed = 0
                for i, (start, end, tok) in enumerate(moves_to_make, 1):
                    print(f"🎯 Executing move {i}: {tok}")
                    make_move(start, end, player, moves_left_ai - i + 1, confidence)
                    moves_executed += 1
                
                return (sequence, moves_executed)

        # Fallback to search AI if no LLM moves work
        print("🔄 All LLM_AI predictions failed - falling back to search_AI for ENTIRE turn")
        global llm_failures
        llm_failures += 1
        return None

    except Exception as e:
        print(f"❌ LLM prediction failed: {e}")
        import traceback
        traceback.print_exc()
        print("⚠️  Falling back to search_AI")
        # DEBUG: LLM system errors - check BackgammonBrain_Plateau_10_31_25.py
        return None


def parse_multi_move_token(move_token, dice, player):
    """
    Legacy support for multi-move tokens if they appear, though atomic is preferred.
    """
    if not move_token.startswith('m_'):
        return None
    
    move_part = move_token[2:]
    
    # Only handle letter-pair format (not 'to' format)
    if 'to' in move_part or len(move_part) % 2 != 0:
        return None
    
    # Extract all move pairs
    move_pairs = []
    for i in range(0, len(move_part), 2):
        start_sym = move_part[i]
        end_sym = move_part[i+1]
        start_pos = letter_to_idx(start_sym, player) if not start_sym.isdigit() else int(start_sym)
        end_pos = letter_to_idx(end_sym, player) if not end_sym.isdigit() else int(end_sym)
        
        if start_pos is None or end_pos is None:
            return None
        
        move_pairs.append((start_pos, end_pos))
    
    if not move_pairs:
        return None
        
    return move_pairs


def token_to_positions(move_token, dice, player):
    """
    === CRITICAL LLM TOKEN PARSING FUNCTION ===
    Convert LLM move token back to board positions (start_pos, end_pos).
    """
    # === BASIC VALIDATION ===
    if not move_token.startswith('m_'):
        print(f"❌ LLM ERROR: Returned '{move_token}' - not a MOVE TOKEN!")
        return None, None

    # Parse the move token
    move_part = move_token[2:]  # Remove "m_" prefix

    # === REGULAR MOVE PARSING ===
    if 'to' in move_part:
        # Regular move: "m_12to15" or letter form "m_gtoj"
        try:
            start_str, end_str = move_part.split('to')
            # Accept both numeric and letter-based positions
            start_pos = int(start_str) if start_str.isdigit() else letter_to_idx(start_str, player)
            end_pos = int(end_str) if end_str.isdigit() else letter_to_idx(end_str, player)
            return start_pos, end_pos
        except (ValueError, IndexError) as e:
            print(f"❌ Token parsing failed: {move_part} ({e})")
            pass

    # === BEARING OFF MOVE PARSING ===
    elif 'off' in move_part:
        # Bearing off move: "m_5off" or letter form "m_goff"
        try:
            start_str = move_part.replace('off', '')
            start_pos = int(start_str) if start_str.isdigit() else letter_to_idx(start_str, player)
            end_pos = 25 if player == -1 else 0  # Bearing off position for correct player
            return start_pos, end_pos
        except (ValueError, IndexError) as e:
            print(f"❌ Bearing off parsing failed: {move_part} ({e})")
            pass
    else:
        # Compact letter token: e.g., m_mk (single move)
        if len(move_part) == 2:
             start_sym = move_part[0]
             end_sym = move_part[1]
             start_pos = letter_to_idx(start_sym, player)
             end_pos = letter_to_idx(end_sym, player)
             return start_pos, end_pos
        elif len(move_part) > 2 and len(move_part) % 2 == 0:
            # Legacy multi-move token support
            # Just return first move
             start_sym = move_part[0]
             end_sym = move_part[1]
             start_pos = letter_to_idx(start_sym, player)
             end_pos = letter_to_idx(end_sym, player)
             return start_pos, end_pos

    return None, None


def move_to_token(start_pos, end_pos, player):
    """
    Convert a board move (start_pos, end_pos) to LLM token format.
    """
    # Emit compact letter tokens for LLM/history while keeping internal state numeric
    if end_pos == 0 or end_pos == 25:  # Bearing off (white=0, black=25)
        return f"m_{idx_to_letter(start_pos)}z"
    else:
        return f"m_{idx_to_letter(start_pos)}{idx_to_letter(end_pos)}"


def add_player_move_to_history(start_pos, end_pos, dice_vals, game_history_tokens):
    """
    Placeholder. Move recording happens in make_move/record_move_pair.
    """
    pass

def evaluate_board(board, bar, player=1):
    score = 0
    
    if player == -1:  # Black player perspective
        # Pip count
        pip_count = 0
        for i in range(1, BOARD_SIZE + 1):
            if board[i] < 0:
                pip_count += abs(board[i]) * (25 - i)  # Black pieces: distance from home (25)
            elif board[i] > 0:
                pip_count -= board[i] * (25 - i)  # White pieces: their progress
        score -= pip_count / 120  # Lower pip count better for black

        # Blots
        blots = sum(1 for i in range(1, BOARD_SIZE + 1) if board[i] == -1)  # Black blots
        score += blots * 100  # Fewer blots better

        # Home board (Red: 19-24, Blue: 1-6)
        black_home_pieces = sum(abs(board[i]) for i in range(19, 25))
        white_home_pieces = sum(board[i] for i in range(1, 7))
        score -= black_home_pieces * 10  # More black home pieces better (reduced weight)
        score += white_home_pieces * 20  # More white home pieces worse

        # Primes
        black_primes = count_primes(board, -1)
        white_primes = count_primes(board, 1)
        score -= black_primes * 100  # More black primes better
        score += white_primes * 50   # More white primes worse

        # Hitting and blocking
        black_hits = sum(1 for i in range(1, BOARD_SIZE + 1) if board[i] == -1)
        white_hits = sum(1 for i in range(1, BOARD_SIZE + 1) if board[i] == 1)
        score += black_hits * 90  # More Red hits better
        score -= white_hits * 40  # More white hits worse

        # Bar
        score += bar[1] * 200  # Black pieces on bar bad (increased penalty)
        score -= bar[0] * 200  # White pieces on bar good (increased bonus)

    else:  # White player perspective (player == 1)
        # Pip count
        pip_count = 0
        for i in range(1, BOARD_SIZE + 1):
            if board[i] > 0:
                pip_count += board[i] * i  # White pieces: distance from home (0)
            elif board[i] < 0:
                pip_count -= abs(board[i]) * i  # Black pieces: their distance from their home
        score -= pip_count / 120  # Lower pip count better for white

        # Blots
        blots = sum(1 for i in range(1, BOARD_SIZE + 1) if board[i] == 1)  # White blots
        score += blots * 100  # Fewer blots better

        # Home board (Blue: 1-6, Red: 19-24)
        white_home_pieces = sum(board[i] for i in range(1, 7))
        black_home_pieces = sum(abs(board[i]) for i in range(19, 25))
        score -= white_home_pieces * 10  # More white home pieces better (reduced weight)
        score += black_home_pieces * 20  # More black home pieces worse

        # Primes
        white_primes = count_primes(board, 1)
        black_primes = count_primes(board, -1)
        score -= white_primes * 100  # More white primes better
        score += black_primes * 50   # More black primes worse

        # Hitting and blocking
        white_hits = sum(1 for i in range(1, BOARD_SIZE + 1) if board[i] == 1)
        black_hits = sum(1 for i in range(1, BOARD_SIZE + 1) if board[i] == -1)
        score += white_hits * 90  # More white hits better
        score -= black_hits * 40  # More Red hits worse

        # Bar
        score += bar[0] * 200  # White pieces on bar bad (increased penalty)
        score -= bar[1] * 200  # Black pieces on bar good (increased bonus)
    
    # Advanced backgammon strategies
    if player == -1:  # Black perspective (home: 19-24)
        # Holding game
        if board[1] >= 2 and board[2] >= 2:
            score -= 200

        # Priming game
        if black_primes >= 2:
            score -= 150

        # Blitz
        black_builders = sum(1 for i in range(1, 7) if board[i] >= 2)
        if black_builders >= 2:
            score -= 130

        # Backgame
        black_back_checkers = sum(abs(board[i]) for i in range(13, 19))
        if black_back_checkers >= 3:
            score -= 80

        # Racing game
        black_lead = sum(i * abs(board[i]) for i in range(1, 7)) - sum(i * board[i] for i in range(19, 25))
        if black_lead > 0:
            score -= black_lead / 4

        # Timing
        timing = sum(1 for i in range(7, 19) if board[i] == -1)
        score -= timing * 50

        # Duplication
        duplication = sum(1 for i in range(7, 19) if board[i] >= 2)
        score -= duplication * 40

        # Diversification
        diversification = sum(1 for i in range(7, 19) if board[i] == -1)
        score += diversification * 20

        # Vulnerability of black pieces
        black_vulnerability = sum(1 for i in range(1, BOARD_SIZE + 1) if board[i] == -1 and is_vulnerable(board, i))
        score += black_vulnerability * 60

        # Attacking potential of white pieces
        white_attacking_potential = sum(1 for i in range(1, BOARD_SIZE + 1) if board[i] >= 2 and has_attacking_potential(board, i))
        score -= white_attacking_potential * 40

        # Connectivity of black pieces
        black_connectivity = evaluate_connectivity(board, -1)
        score -= black_connectivity * 30

        # Mobility of black pieces
        black_mobility = evaluate_mobility(board, -1)
        score -= black_mobility * 20

    else:  # White perspective (home: 1-6)
        # Holding game
        if board[24] >= 2 and board[23] >= 2:
            score -= 200

        # Priming game
        if white_primes >= 2:
            score -= 150

        # Blitz
        white_builders = sum(1 for i in range(19, 25) if board[i] >= 2)
        if white_builders >= 2:
            score -= 130

        # Backgame
        white_back_checkers = sum(board[i] for i in range(7, 13))
        if white_back_checkers >= 3:
            score -= 80

        # Racing game
        white_lead = sum(i * board[i] for i in range(19, 25)) - sum(i * abs(board[i]) for i in range(1, 7))
        if white_lead > 0:
            score -= white_lead / 4

        # Timing
        timing = sum(1 for i in range(7, 19) if board[i] == 1)
        score -= timing * 50

        # Duplication
        duplication = sum(1 for i in range(7, 19) if board[i] >= 2)
        score -= duplication * 40

        # Diversification
        diversification = sum(1 for i in range(7, 19) if board[i] == 1)
        score += diversification * 20

        # Vulnerability of white pieces
        white_vulnerability = sum(1 for i in range(1, BOARD_SIZE + 1) if board[i] == 1 and is_vulnerable_white(board, i))
        score += white_vulnerability * 60

        # Attacking potential of black pieces
        black_attacking_potential = sum(1 for i in range(1, BOARD_SIZE + 1) if board[i] <= -2 and has_attacking_potential_black(board, i))
        score -= black_attacking_potential * 40

        # Connectivity of white pieces
        white_connectivity = evaluate_connectivity(board, 1)
        score -= white_connectivity * 30

        # Mobility of white pieces
        white_mobility = evaluate_mobility(board, 1)
        score -= white_mobility * 20
    
    return score


def is_vulnerable(board, position):
    """Check if a black piece at the given position is vulnerable to capture."""
    if position <= 1 or position >= BOARD_SIZE:
        return False
    return board[position] == -1 and board[position + 1] >= 1


def is_vulnerable_white(board, position):
    """Check if a white piece at the given position is vulnerable to capture."""
    if position <= 1 or position >= BOARD_SIZE:
        return False
    return board[position] == 1 and board[position - 1] <= -1


def has_attacking_potential(board, position):
    """Check if a white piece at the given position has attacking potential."""
    if position <= 1 or position > BOARD_SIZE:
        return False
    return board[position] >= 2 and board[position - 1] <= 0


def has_attacking_potential_black(board, position):
    """Check if a black piece at the given position has attacking potential."""
    if position < 1 or position >= BOARD_SIZE:
        return False
    return board[position] <= -2 and board[position + 1] >= 0


def evaluate_connectivity(board, player):
    """Evaluate the connectivity of pieces for the given player."""
    connectivity = 0
    for i in range(1, BOARD_SIZE):
        if board[i] * player > 0 and board[i + 1] * player > 0:
            connectivity += 1
    return connectivity


def evaluate_mobility(board, player):
    """Evaluate the mobility of pieces for the given player."""
    mobility = 0
    for i in range(1, BOARD_SIZE + 1):
        if board[i] * player > 0:
            if i > 1 and board[i - 1] * player >= 0:
                mobility += 1
            if i < BOARD_SIZE and board[i + 1] * player >= 0:
                mobility += 1
    return mobility

def count_primes(board, player):
    primes = 0
    consecutive = 0
    for i in range(1, BOARD_SIZE + 1):
        if board[i] * player >= 2:
            consecutive += 1
            if consecutive >= 6:
                primes += 1
        else:
            consecutive = 0
    return primes

def get_clicked_position(pos):
    x, y = pos
    #Want to click at center to move off board (or bar)
    # Much smaller detection area around the bar line to avoid interfering with board positions
    bar_detection_width = 20  # Small area around the bar line
    if x > WIDTH // 2 - bar_detection_width and x < WIDTH // 2 + bar_detection_width :
        if bar[0] > 0:
            return 25  # White bar
        return 0  # Black bar
    if y < HEIGHT // 2:
        # Top row: 13-24 (left to right)
        return 13 + x // (WIDTH // 12)
    else:
        # Bottom row: 1-12 (right to left)
        return 12 - x // (WIDTH // 12)

def is_valid_move(start, end, dice, player, board=board, bar=bar):
    # Check if the move is for the white player
    #print(f"start: {start}, end: {end}, dice: {dice}, player: {player}")
    if player == 1:  # White player
        # Handle moves from the bar
        if bar[0] > 0:
            if start != 25:  # Ensure the move starts from the bar (white bar = 25)
                #print("Invalid selection: White must move from the bar.")
                return False
            if end < 1 or end > BOARD_SIZE or board[end] < -1:
                #print("Invalid move: Cannot move to this position from the bar.")
                return False
            # Check if the move matches one of the dice values
            return (dice[0] > 0 and dice[0] < 7 and end == 25 - dice[0]) or (len(dice) > 1 and dice[1] > 0 and dice[1] < 7 and end == 25 - dice[1]) or (len(dice) > 2 and dice[2] > 0 and dice[2] < 7 and end == 25 - dice[2]) or (len(dice) > 3 and dice[3] > 0 and dice[3] < 7 and end == 25 - dice[3])
        # Handle regular moves
        if start < 1 or start > BOARD_SIZE or board[start] <= 0:
            #print("Invalid selection: No white piece at this position.")
            return False
        if end < 0 or end > BOARD_SIZE or (end >= 1 and board[end] < -1):
            #print("Invalid move: Cannot move to this position.")
            return False

        # Check if moving a piece off the board (white bears off to 0)
        if end == 0:
            # All white pieces must be in home board (7-24 empty)
            if not all(board[i] <= 0 for i in range(BLUE_HOME_END + 1, BOARD_SIZE + 1)):
                return False

            # Must bear off from exact die matches first, then furthest positions
            exact_matches = []
            other_matches = []

            for pos in range(1, BLUE_HOME_END + 1):
                if board[pos] > 0:  # Has white piece
                    distance = pos
                    for d in dice:
                        if d > 0 and d < 7:
                            if d == distance:  # Exact match - highest priority
                                exact_matches.append(pos)
                            elif d > distance:  # Can use higher die for this position
                                other_matches.append(pos)

            if exact_matches:
                # Must bear off from exact match positions
                return start in exact_matches
            elif other_matches:
                # Can bear off from furthest position (highest number in home board)
                furthest_pos = max(other_matches)
                return start == furthest_pos
            else:
                return False

            return False
        
        # Check if the move matches one of the dice values for regular moves
        # White moves counter-clockwise (decreasing positions)
        return (dice[0] > 0 and dice[0] < 7 and start - end == dice[0]) or (len(dice) > 1 and dice[1] > 0 and dice[1] < 7 and start - end == dice[1]) or (len(dice) > 2 and dice[2] > 0 and dice[2] < 7 and start - end == dice[2]) or (len(dice) > 3 and dice[3] > 0 and dice[3] < 7 and start - end == dice[3])
    
    else:  # Black player
        # Similar logic for the black player with adjustments for direction and starting positions
        if bar[1] > 0:
            if start != -1:  # Black bar = -1 (consistent with letter_to_idx)
                #print("Invalid selection: Black must move from the bar.")
                return False
            if end < 1 or end > BOARD_SIZE or board[end] > 1:
                #print("Invalid move: Cannot move to this position from the bar.")
                return False
            return (dice[0] > 0 and dice[0] < 7 and end == dice[0]) or (len(dice) > 1 and dice[1] > 0 and dice[1] < 7 and end == dice[1]) or (len(dice) > 2 and dice[2] > 0 and dice[2] < 7 and end == dice[2]) or (len(dice) > 3 and dice[3] > 0 and dice[3] < 7 and end == dice[3])
        if start < 1 or start > BOARD_SIZE or board[start] >= 0:
            #print("Invalid selection: No black piece at this position.")
            return False
        if end < 1 or end > 25 or (end <= BOARD_SIZE and board[end] > 1):
            #print("Invalid move: Cannot move to this position.")
            return False
        
        if end == 25:
            # All black pieces must be in home board (1-18 empty)
            if not all(board[i] >= 0 for i in range(1, RED_HOME_START)):
                return False

            # Must bear off from exact die matches first, then furthest positions
            exact_matches = []
            other_matches = []

            for pos in range(RED_HOME_START, BOARD_SIZE + 1):
                if board[pos] < 0:  # Has black piece
                    distance = 25 - pos
                    for d in dice:
                        if d > 0 and d < 7:
                            if d == distance:  # Exact match - highest priority
                                exact_matches.append(pos)
                            elif d > distance:  # Can use higher die for this position
                                other_matches.append(pos)

            if exact_matches:
                # Must bear off from exact match positions
                return start in exact_matches
            elif other_matches:
                # Can bear off from furthest position (lowest number in home board)
                furthest_pos = min(other_matches)
                return start == furthest_pos
            else:
                return False

            return False
        # Check if the move matches one of the dice values for regular moves
        # Black moves clockwise (increasing positions)
        return (dice[0] > 0 and dice[0] < 7 and end - start == dice[0]) or (len(dice) > 1 and dice[1] > 0 and dice[1] < 7 and end - start == dice[1]) or (len(dice) > 2 and dice[2] > 0 and dice[2] < 7 and end - start == dice[2]) or (len(dice) > 3 and dice[3] > 0 and dice[3] < 7 and end - start == dice[3]) 


def has_valid_moves(player, dice, board=board, bar=bar):
    for start in range(-1 if player == -1 else 1, (BOARD_SIZE + 2 if player == 1 else BOARD_SIZE + 1)):
        if start == -1 and (bar[1] == 0 if player == -1 else True):
            continue
        if start == 25 and (bar[0] == 0 if player == 1 else True):
            continue
        # Check if the current player has pieces at this position
        if start >= 1 and start <= BOARD_SIZE:
            if player == 1 and board[start] <= 0:  # White pieces are positive
                continue
            if player == -1 and board[start] >= 0:  # Black pieces are negative
                continue
        for end in range(0, BOARD_SIZE + 2):
            if is_valid_move(start, end, dice, player, board, bar):
                return True
    return False      

    
# Function to make a move on the board
def make_move(start, end, player, moves_left=0, score=0):
    global white_pieces_off, black_pieces_off, dice1, dice2, dice3, dice4, bar
    font = pygame.font.Font(None, 32)

    # === STEP 1: Highlight FROM position before moving ===
    from_positions = []
    if start >= 1 and start <= BOARD_SIZE:
        from_positions.append(start)
    elif start == 25:  # White bar - highlight all white bar positions
        from_positions = [25]  # We'll handle bar highlighting specially
    elif start == -1:  # Black bar - highlight all black bar positions
        from_positions = [-1]  # We'll handle bar highlighting specially

    if from_positions:
        draw_board(board, bar, from_positions)
        draw_dice(dice1, dice2)
        pygame.display.flip()
        time.sleep(0.33)  # Show FROM highlight for 1/3 second

    # === STEP 2: Execute the actual move ===
    if player == 1:  # White player
        if start == 25:  # White bar
            bar[0] -= 1
        else:
            board[start] -= 1
        if end == 0:  # Blue bears off to 0
            print(f"Blue's move: {idx_to_letter(start)}({start}) off the board")
            white_pieces_off += 1  # Increment the count of blue pieces off the board
            text = font.render (f"Blue: {idx_to_letter(start)}({start}) to off", True, BLACK)
            # Clear the text area before drawing new text
            text_rect = pygame.Rect(10, HEIGHT + 25 * (4-moves_left)+50, 300, 82)
            pygame.draw.rect(screen, WHITE_ISH, text_rect)
            screen.blit(text, (10, HEIGHT + 25 * (4-moves_left)+50))
        elif board[end] == -1:
            board[end] = 1
            bar[1] += 1
            print(f"White hits: {idx_to_letter(start)}({start}) to {idx_to_letter(end)}({end}), moves left: {moves_left}")
            text = font.render (f"White hits: {idx_to_letter(start)}({start}) to {idx_to_letter(end)}({end})", True, BLACK)
            # Clear the text area before drawing new text
            text_rect = pygame.Rect(10, HEIGHT + 25 * (4-moves_left)+50, 300, 82)
            pygame.draw.rect(screen, WHITE_ISH, text_rect)
            screen.blit(text, (10, HEIGHT + 25 * (4-moves_left)+50))
        else:
            board[end] += 1
            print(f"Blue's move: {idx_to_letter(start)}({start}) to {idx_to_letter(end)}({end}), moves left: {moves_left}")
            text = font.render (f"Blue: {idx_to_letter(start)}({start}) to {idx_to_letter(end)}({end})", True, BLACK)
            # Clear the text area before drawing new text
            text_rect = pygame.Rect(10, HEIGHT + 25 * (4-moves_left)+50, 300, 82)
            pygame.draw.rect(screen, WHITE_ISH, text_rect)
            screen.blit(text, (10, HEIGHT + 25 * (4-moves_left)+50))

    else:  # Black player (AI)
        if start == -1:  # Black bar
            bar[1] -= 1
        else:
            board[start] += 1
        if end == 25:  # Red bears off to 25
            print(f"Red's move: {idx_to_letter(start)}({start}) off the board")
            black_pieces_off += 1  # Increment the count of red pieces off the board
            text = font.render (f"Red: {idx_to_letter(start)}({start}) off  score {score}", True, BLACK)
            # Clear the text area before drawing new text
            text_rect = pygame.Rect(WIDTH//2 + 10, HEIGHT + 25 * (4-moves_left)+50, 350, 32)
            pygame.draw.rect(screen, WHITE_ISH, text_rect)
            screen.blit(text, (WIDTH//2 + 10, HEIGHT + 25 * (4-moves_left)+50))
        elif board[end] == 1:
            board[end] = -1
            bar[0] += 1
            print(f"Red hits: {idx_to_letter(start)}({start}) to {idx_to_letter(end)}({end}), score {score}, moves left: {moves_left}")
            text = font.render (f"Red hits: {idx_to_letter(start)}({start}) to {idx_to_letter(end)}({end})  score {int(score)}", True, BLACK)
            # Clear the text area before drawing new text
            text_rect = pygame.Rect(WIDTH//2 + 10, HEIGHT + 25 * (4-moves_left)+50, 350, 32)
            pygame.draw.rect(screen, WHITE_ISH, text_rect)
            screen.blit(text, (WIDTH//2 + 10, HEIGHT + 25 * (4-moves_left)+50))
        else:
            board[end] -= 1
            print(f"Red's move: {idx_to_letter(start)}({start}) to {idx_to_letter(end)}({end}), moves left: {moves_left}, Score: {score}")
            text = font.render (f"Red: {idx_to_letter(start)}({start}) to {idx_to_letter(end)}({end})  score {int(score)}", True, BLACK)
            # Clear the text area before drawing new text
            text_rect = pygame.Rect(WIDTH//2 + 10, HEIGHT + 25 * (4-moves_left)+50, 350, 32)
            pygame.draw.rect(screen, WHITE_ISH, text_rect)
            screen.blit(text, (WIDTH//2+ 10, HEIGHT + 25 * (4-moves_left)+50))

    # === STEP 3: Highlight TO position after moving ===
    to_positions = []
    if end >= 1 and end <= BOARD_SIZE:
        to_positions.append(end)

    if to_positions:
        draw_board(board, bar, to_positions)
        draw_dice(dice1, dice2)
        pygame.display.flip()
        time.sleep(0.33)  # Show TO highlight for 1/3 second
        draw_board(board, bar)  # Back to normal
        draw_dice(dice1, dice2)

    pygame.display.flip()
    # Record pair for single-turn combined token
    record_move_pair(start, end, player)
    # Update used dice
    if end == 25 or end == 0:  # Bearing off
        # For bearing off, find which die was used by checking all possible valid dice
        distance = 25 - start if player == -1 else start
        # Check all dice to see which one could have been used for this bearing off
        for d in [dice1, dice2, dice3, dice4]:
            if d > 0 and d < 7:
                # Check if this die could be used for bearing off
                if d == distance:  # Exact match
                    move_distance = d
                    break
                elif d > distance:  # Higher die - check if no closer pieces
                    has_closer_pieces = False
                    if player == -1:  # Black
                        has_closer_pieces = any(board[i] < 0 for i in range(start + 1, BOARD_SIZE + 1))
                    else:  # White
                        has_closer_pieces = any(board[i] > 0 for i in range(1, start))
                    if not has_closer_pieces:
                        move_distance = d
                        break
        else:
            move_distance = 0  # No valid die found (shouldn't happen)
    else:
        # Handle bar moves correctly - die value equals distance from bar
        if start == 25:  # White bar
            move_distance = 25 - end
        elif start == -1:  # Black bar
            move_distance = end
        else:  # Regular board moves
            move_distance = abs(end - start)
    if dice1 == move_distance:
        dice1 = 7
    elif dice2 == move_distance:
        dice2 = 7
    elif dice3 == move_distance:
        dice3 = 0
    elif dice4 == move_distance:
        dice4 = 0
    
    
# Function to check if the game is over
def is_game_over():
    return white_pieces_off == 15 or black_pieces_off == 15


# Function to display the current player's turn
def display_turn(player_turn):
    font = pygame.font.Font(None, 32)
    if player_turn == 1:
        if versus_mode:
            text = font.render("Blue AI rolling...", True, BLACK)
        else:
            text = font.render("Blue's Turn roll Die", True, BLACK)
        x = 10
    else:
        ai_name = ai_type.replace("_", " ").upper()
        if versus_mode:
            text = font.render(f"Red {ai_name} Thinking..", True, BLACK)
        else:
            text = font.render(f"Red {ai_name} Thinking..", True, BLACK)
        x = 410

    # Clear the text area before displaying
    text_rect = pygame.Rect(x, HEIGHT + 25, 400, 32)
    pygame.draw.rect(screen, WHITE_ISH, text_rect)
    screen.blit(text, (x, HEIGHT + 25))

    # Clear and show versus mode indicator
    vs_rect = pygame.Rect(WIDTH - 120, HEIGHT + 10, 120, 20)
    pygame.draw.rect(screen, WHITE_ISH, vs_rect)
    if versus_mode:
        font_small = pygame.font.Font(None, 20)
        vs_text = font_small.render("VERSUS MODE", True, (255, 0, 0))
        screen.blit(vs_text, (WIDTH - 120, HEIGHT + 10))

# Function to display the AI thinking message
def time_ai_thinking():
    time.sleep(0.5)  # Reduced time for LLM testing

# Function to display the winner
def display_winner(winner):
    # Clear the winner display area
    winner_rect = pygame.Rect(0, HEIGHT + 70, WIDTH, 120)
    pygame.draw.rect(screen, WHITE_ISH, winner_rect)

    font = pygame.font.Font(None, 48)
    text = font.render(f"{winner} wins!", True, BLACK)
    font_small = pygame.font.Font(None, 24)
    restart_text = font_small.render("Press R to restart or Q to quit", True, BLACK)

    screen.blit(text, (10, HEIGHT + 80))
    screen.blit(restart_text, (10, HEIGHT + 130))
    pygame.display.flip()
    # Don't wait - let user choose restart or quit

# ============================================================================
# LLM INTEGRATION FOR BACKGAMMON AI
# ============================================================================
# This program supports two AI types:
# 1. search_AI: Traditional minimax algorithm with position evaluation
# 2. LLM_AI: Neural network model trained on expert backgammon games
#
# LLM Integration Features:
# - Full game history tracking (like chess implementation)
# - Strategic move prediction based on complete context
# - Top-5 move suggestions with confidence scores
# - Automatic fallback to search_AI for illegal moves
# - Console logging for transparency
# ============================================================================

# AI Selection
ai_type = "LLM_AI"  # Default to LLM AI
llm_predictor = None

if LLM_AVAILABLE:
    print("\n🤖 AI Selection:")
    print("1. search_AI (current minimax algorithm)")
    print("2. LLM_AI (neural network model)")
    while True:
        try:
            choice = input("Select AI type (1 or 2, default=2): ").strip()
            if choice == "1":
                ai_type = "search_AI"
                print("Selected: search_AI")
                break
            elif choice == "2" or choice == "":  # Empty input defaults to LLM_AI
                ai_type = "LLM_AI"
                print("Selected: LLM_AI" if choice == "2" else "Selected: LLM_AI (default)")
                try:
                    llm_predictor = BackgammonMovePredictor()
                    print("✅ LLM predictor initialized successfully!")
                except Exception as e:
                    print(f"❌ Failed to initialize LLM predictor: {e}")
                    print("Falling back to search_AI")
                    ai_type = "search_AI"
                break
            else:
                print("Please enter 1 or 2 (or press Enter for default)")
        except KeyboardInterrupt:
            ai_type = "LLM_AI"
            print("\nSelected: LLM_AI (default)")
            break
else:
    print("LLM_AI not available, using search_AI")

# === MAIN GAME LOOP INITIALIZATION ===
running = True
game_over = False
versus_mode = False  # When True, AI plays both sides

# === CRITICAL GAME STATE VARIABLES ===
player_turn = 1  # 1 = white (human player), -1 = black (AI)
selected_piece = None  # Which piece the player has clicked on (None = no selection)

# === DICE STATE ===
dice_rolled = False  # Whether dice have been rolled for current turn
dice1, dice2 = 0, 0  # Current dice values
dice3, dice4 = 0, 0  # Extra dice for doubles (when dice1 == dice2)
moves_left = 0       # How many moves the current player has left

# === INITIAL DIE ROLL STATE (for determining who goes first) ===
initial_roll_done = False  # Whether initial die roll has been completed
player_initial_die = 0     # Player's initial die roll (0 = not rolled)
ai_initial_die = 0         # AI's initial die roll (0 = not rolled)

# === UI SETUP ===
font = pygame.font.Font(None, 32)
rect = pygame.Rect(0, HEIGHT, WIDTH, HEIGHT+100)  # Bottom UI area
pygame.draw.rect(screen, WHITE_ISH, rect)

# === LLM CONTEXT TRACKING (CRITICAL FOR AI FUNCTIONALITY) ===
# The LLM needs the COMPLETE game history to make strategic decisions
# Format: ["<STARTGAME>", "d31", "m_0to3", "d52", "m_3to8", ...]
# - "d{dice}" = dice roll (e.g., "d31" = rolled 3 and 1)
# - "m_{move}" = move token (e.g., "m_0to3" = move from position 0 to 3)
# Both human and AI moves are tracked to maintain accurate context
# DEBUG: Check game_history_tokens to see full game state for LLM
game_history_tokens = ["<STARTGAME>"]
player_pairs_accum = []  # holds letter pairs for current white turn
ai_pairs_accum = []      # holds letter pairs for current black turn
current_player_dice = None  # holds current player's dice token until turn completion
current_ai_dice = None      # holds current AI's dice token until turn completion

def record_move_pair(start_pos, end_pos, player):
    start_l = idx_to_letter(start_pos)
    end_l = idx_to_letter(end_pos)
    pair = f"{start_l}{end_l if end_l is not None else ''}"
    if player == 1:
        player_pairs_accum.append(pair)
    else:
        ai_pairs_accum.append(pair)

def flush_turn_history(player):
    global current_player_dice, current_ai_dice

    # Add dice token first (if any)
    # ATOMIC CHANGE: split dice string into atomic tokens
    dice_to_add = None
    if player == 1 and current_player_dice:
        dice_to_add = current_player_dice
        current_player_dice = None
    elif player == -1 and current_ai_dice:
        dice_to_add = current_ai_dice
        current_ai_dice = None
        
    if dice_to_add:
        # Handle dice token splitting
        # Format expected from game logic: "d66" or "d41"
        if dice_to_add.startswith("d"):
            raw_dice = dice_to_add[1:] # "66"
            if len(raw_dice) == 2:
                tokens = [f"d{raw_dice[0]}", f"d{raw_dice[1]}"]
                game_history_tokens.extend(tokens)
                print(f"📝 Added dice tokens to history: {tokens}")
            else:
                # Fallback/Shouldn't happen
                game_history_tokens.append(dice_to_add)
                print(f"📝 Added dice token to history: {dice_to_add}")
    
    # Then add move tokens
    # ATOMIC CHANGE: Convert pair strings to atomic move tokens
    if player == 1 and player_pairs_accum:
        # player_pairs_accum is list like ['ab', 'cd']
        for pair in player_pairs_accum:
            token = "m_" + pair
            game_history_tokens.append(token)
        game_history_tokens.append("<EOM>")
        print(f"📝 Added player turn tokens to history: {player_pairs_accum} + <EOM>")
        player_pairs_accum.clear()
        
    elif player == -1 and ai_pairs_accum:
        for pair in ai_pairs_accum:
            token = "m_" + pair
            game_history_tokens.append(token)
        game_history_tokens.append("<EOM>")
        print(f"📝 Added AI turn tokens to history: {ai_pairs_accum} + <EOM>")
        ai_pairs_accum.clear()
    elif dice_to_add: # Dice rolled but no moves?
         game_history_tokens.append("<NOMOVE>")
         game_history_tokens.append("<EOM>")
         print("📝 Added <NOMOVE> <EOM> to history")

display_turn(player_turn)

# === INITIAL BOARD DISPLAY ===
# Draw the board for the first time so player can see it during die rolls
draw_board(board, bar)
pygame.display.flip()

# === INITIAL DIE ROLL TO DETERMINE WHO GOES FIRST ===
# Wait for player to press space to roll
print("🎲 Waiting for player to press space to roll dice...")
display_roll_prompt()

waiting_for_initial_roll = True
while waiting_for_initial_roll and running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
            waiting_for_initial_roll = False
        elif event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
            # Roll single die for each player to determine turn order
            player_initial_die = random.randint(1, 6)
            ai_initial_die = random.randint(1, 6)
            display_initial_rolls(player_initial_die, ai_initial_die)
            waiting_for_initial_roll = False
            break
    pygame.time.wait(50)  # Small delay to prevent excessive CPU usage

# Handle ties - keep rolling until no tie
while player_initial_die == ai_initial_die and running:
    print(f"Tie! Player rolled {player_initial_die}, AI rolled {ai_initial_die}")
    waiting_for_space = True
    while waiting_for_space and running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                waiting_for_space = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                player_initial_die = random.randint(1, 6)
                ai_initial_die = random.randint(1, 6)
                display_initial_rolls(player_initial_die, ai_initial_die)
                waiting_for_space = False
                break
        pygame.time.wait(50)  # Small delay to prevent excessive CPU usage

# Set player_turn based on who rolled higher and use both dice from the roll
if player_initial_die > ai_initial_die:
    player_turn = 1  # Player (red/white) goes first
    # Winner uses BOTH dice from the roll (sorted higher first)
    dice1 = max(player_initial_die, ai_initial_die)
    dice2 = min(player_initial_die, ai_initial_die)
    # Set moves_left for player turn
    moves_left = 2
    if dice1 == dice2:
        moves_left *= 2  # Doubles = 4 moves
        dice3 = dice1
        dice4 = dice2
    print(f"Player rolled {player_initial_die}, AI rolled {ai_initial_die} - Player goes first!")
    print(f"First turn dice: {dice1}, {dice2}")
elif ai_initial_die > player_initial_die:
    player_turn = -1  # AI (blue/black) goes first
    # Winner uses BOTH dice from the roll (sorted higher first)
    dice1 = max(player_initial_die, ai_initial_die)
    dice2 = min(player_initial_die, ai_initial_die)
    # Set moves_left_ai for AI turn
    moves_left_ai = 2
    if dice1 == dice2:
        moves_left_ai = 4  # Doubles = 4 moves
        dice3 = dice1
        dice4 = dice2
    print(f"Player rolled {player_initial_die}, AI rolled {ai_initial_die} - AI goes first!")
    print(f"First turn dice: {dice1}, {dice2}")

# Track the initial roll for LLM context
initial_dice_token = f"d{dice1}{dice2}"  # Format: d53 for dice 5,3
game_history_tokens.append(initial_dice_token)
print(f"📝 Added initial dice token to history: {initial_dice_token}")

initial_roll_done = True
dice_rolled = True  # Set dice as rolled so the game can proceed
pygame.time.wait(2000)  # Show the result for 2 seconds before starting game

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if not versus_mode and player_turn == 1 and dice_rolled:
                pos = pygame.mouse.get_pos()
                clicked_position = get_clicked_position(pos)
                print(f"Clicked position: {clicked_position}")
                
                if selected_piece is None:
                    if clicked_position == 25 and bar[0] > 0:  # White bar
                        selected_piece = 25
                    elif clicked_position <= BOARD_SIZE and board[clicked_position] > 0:
                        selected_piece = clicked_position
                        print(f"Selected piece: {selected_piece}")
                        text = font.render(f"Selected piece: {selected_piece}", True, BLACK)
                        rect = pygame.Rect(0, HEIGHT+150, WIDTH, 50)
                        pygame.draw.rect(screen, WHITE_ISH, rect)
                        screen.blit(text, (10, HEIGHT + 150))
                        
                    else:
                        selected_piece = None
                        print("Invalid selection for white player. Please select a valid piece.")
                        text = font.render("Invalid selection for white player. Please select a valid piece.", True, BLACK)
                        rect = pygame.Rect(0, HEIGHT+150, WIDTH, 50)
                        pygame.draw.rect(screen, WHITE_ISH, rect)
                        screen.blit(text, (10, HEIGHT + 150))
                else:
                    if clicked_position == selected_piece:
                        selected_piece = None
                        print("Deselected")
                        text = font.render("Deselected", True, BLACK)
                        rect = pygame.Rect(0, HEIGHT+150, WIDTH, 50)
                        pygame.draw.rect(screen, WHITE_ISH, rect)
                        screen.blit(text, (10, HEIGHT + 150))
                    else:
                        print(f"Clicked move to: {clicked_position}")
                        text = font.render(f"Clicked move to: {clicked_position}", True, BLACK)
                        rect = pygame.Rect(0, HEIGHT+150, WIDTH, 50)
                        pygame.draw.rect(screen, WHITE_ISH, rect)
                        screen.blit(text, (10, HEIGHT + 150))
                        if is_valid_move(selected_piece, clicked_position, [dice1, dice2], 1):
                            make_move(selected_piece, clicked_position, 1, moves_left)

                            # Add player move to game history for LLM context
                            # Note: dice token was already added when dice were rolled
                            add_player_move_to_history(selected_piece, clicked_position, [dice1, dice2], game_history_tokens)

                            if dice1 ==7 and dice3:
                                dice1 = dice3
                                dice3 = 0
                            if dice2 ==7 and dice4:
                                dice2 = dice4
                                dice4 = 0
                            selected_piece = None
                            moves_left -= 1

                            # Check if turn should end early (no more legal moves with remaining dice)
                            remaining_dice = []
                            if dice1 > 0 and dice1 < 7: remaining_dice.append(dice1)
                            if dice2 > 0 and dice2 < 7: remaining_dice.append(dice2)
                            if dice3 > 0 and dice3 < 7: remaining_dice.append(dice3)
                            if dice4 > 0 and dice4 < 7: remaining_dice.append(dice4)

                            if moves_left > 0 and remaining_dice and not has_valid_moves(1, remaining_dice):
                                # No more legal moves with remaining dice - end turn early
                                print(f"No more legal moves with remaining dice {remaining_dice}. Ending turn early.")
                                moves_left = 0

                            draw_board(board, bar)
                            draw_dice(dice1, dice2)
                            pygame.display.flip()

                            if moves_left == 0:
                                # Flush a single combined move token for the player's turn
                                flush_turn_history(1)
                                player_turn = -1
                                display_turn(player_turn)
                                pygame.display.flip()
                                dice_rolled = False
                                dice3, dice4 = 0, 0
                                if is_game_over():
                                    # Winner is whoever got 15 pieces off first
                                    if white_pieces_off == 15:
                                        winner = "Blue"
                                    elif black_pieces_off == 15:
                                        winner = "Red"
                                    else:
                                        winner = None  # Should not happen
                                    print(f"🎉 GAME OVER! {winner} wins!")
                                    print(f"🤖 LLM AI failures during this game: {llm_failures}")
                                    display_winner(winner)
                                    game_over = True
                        else:
                            selected_piece = None
                            print("Invalid move, please select another piece.")
                            text = font.render("Invalid move, please select another piece.", True, BLACK)
                            rect = pygame.Rect(0, HEIGHT+150, WIDTH, 50)
                            pygame.draw.rect(screen, WHITE_ISH, rect)
                            screen.blit(text, (10, HEIGHT + 150))

        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_v:
                # Toggle versus mode
                versus_mode = not versus_mode
                if versus_mode:
                    print("🤖🤖 VERSUS MODE: AI vs AI enabled!")
                else:
                    print("👤🤖 HUMAN MODE: Human vs AI enabled!")
            elif event.key == pygame.K_SPACE and player_turn == 1 and not dice_rolled:
                # Clear move text areas from previous turn
                for i in range(4):
                    move_rect = pygame.Rect(10, HEIGHT + 25 * (4 - (i + 1)) + 50, 300, 32)
                    pygame.draw.rect(screen, WHITE_ISH, move_rect)

                dice1, dice2 = roll_dice()
                print ("Blue rolled:",dice1, dice2)

                # Store dice token for later addition to history (after turn completion)
                # Use canonical order: larger die first (matches LLM training data)
                if dice1 >= dice2:
                    current_player_dice = f"d{dice1}{dice2}"
                else:
                    current_player_dice = f"d{dice2}{dice1}"
                print(f"📝 Player dice rolled: {current_player_dice} (will add to history after turn)")

                rect = pygame.Rect(0, HEIGHT, WIDTH//2, 200)
                pygame.draw.rect(screen, WHITE_ISH, rect)
                draw_dice(dice1, dice2)
                dice_rolled = True
                moves_left = 2
                if dice1 == dice2:
                    moves_left *= 2
                    dice3 = dice1
                    dice4 = dice2
                didnt_validate = True

                # Check if human player has any valid moves
                dice_list = [dice1, dice2]
                if moves_left == 4:
                    dice_list.extend([dice3, dice4])
                if not has_valid_moves(1, dice_list):
                    print(f"No valid moves for human player with dice {dice1} {dice2}. Turn passes.")
                    # Clear the message area first
                    message_rect = pygame.Rect(10, HEIGHT+150, 500, 30)
                    pygame.draw.rect(screen, WHITE_ISH, message_rect)
                    text = font.render(f"You have no valid moves with {dice1} {dice2}. Turn passes.", True, BLACK)
                    screen.blit(text, (10, HEIGHT+150))
                    pygame.display.flip()
                    pygame.time.wait(2000)  # Show message for 2 seconds
                    # Add dice to history since turn is passing
                    # ATOMIC UPDATE handled in flush (dice is set in current_player_dice)
                    flush_turn_history(1) # This will add dice and NOMOVE/EOM if no pairs accumulated
                    
                    # Switch to AI turn
                    player_turn = -1
                    dice_rolled = False
                    continue

    # Validation now happens after each move - removed buggy validation here
    
    if (player_turn == -1 or (versus_mode and player_turn == 1)) and not dice_rolled and not game_over:
        # Clear move text areas from previous turn
        for i in range(4):
            move_rect = pygame.Rect(10, HEIGHT + 25 * (4 - (i + 1)) + 50, 300, 32)
            pygame.draw.rect(screen, WHITE_ISH, move_rect)

        display_turn(player_turn)
        dice1, dice2 = roll_dice()
        #dice1, dice2 = 4,4 # test
        draw_dice(dice1, dice2)
        print ("Red AI rolled:",dice1, dice2)

        # Store dice token for later addition to history (after turn completion)
        # Use canonical order: larger die first (matches LLM training data)
        if dice1 >= dice2:
            dice_token = f"d{dice1}{dice2}"
        else:
            dice_token = f"d{dice2}{dice1}"

        # Set the correct dice token variable based on whose turn it is
        if player_turn == 1:
            current_player_dice = dice_token
            print(f"📝 Player dice rolled: {current_player_dice} (will add to history after turn)")
        else:
            current_ai_dice = dice_token
            print(f"📝 AI dice rolled: {current_ai_dice} (will add to history after turn)")

        font = pygame.font.Font(None, 36)
        rect = pygame.Rect(WIDTH//2, HEIGHT+50, WIDTH//2, 150)
        pygame.draw.rect(screen, WHITE_ISH, rect)
        pygame.display.flip()
        time_ai_thinking()
        dice_rolled = True
        moves_left_ai = 2
        if dice1 == dice2:
            dice3 = dice1
            dice4 = dice2
            moves_left_ai = 4


    # === AI TURN EXECUTION ===
    # This is where the AI makes its moves. Supports both LLM and traditional search AI
    if (player_turn == -1 or (versus_mode and player_turn == 1)) and dice_rolled and not game_over:
        
        # === LLM AI PRIMARY PATH ===
        # LLM is called ONCE per turn with fresh dice and predicts entire turn
        if ai_type == "LLM_AI":
            # Call LLM once at start of turn with fresh dice
            llm_result = llm_ai_move([dice1, dice2, dice3, dice4], moves_left_ai, game_history_tokens, player_turn)
            
            # Check if LLM returned a multi-move result (token, num_moves) or single move (token) or None
            if llm_result is None:
                # LLM failed completely, use search AI for ENTIRE turn
                print("🔄 LLM_AI failed, using search_AI for ENTIRE turn (all remaining moves)")
                moves_made = 0
                while moves_left_ai > 0:
                    search_move_made = perform_search_ai_with_move_capture([dice1, dice2, dice3, dice4], moves_left_ai, player_turn)
                    if search_move_made:
                        moves_left_ai -= 1
                        moves_made += 1
                    else:
                        break
                    if versus_mode:
                        pygame.time.wait(500)

                # If no moves were made by either LLM or search_AI, record dice with no moves
                if moves_made == 0:
                    print(f"No valid moves for AI with dice {dice1} {dice2}. Turn passes.")
                    # Clear the message area first
                    message_rect = pygame.Rect(10, HEIGHT+150, 500, 30)
                    pygame.draw.rect(screen, WHITE_ISH, message_rect)
                    text = font.render(f"AI has no valid moves with {dice1} {dice2}. Turn passes.", True, BLACK)
                    screen.blit(text, (10, HEIGHT+150))
            else:
                # LLM succeeded - moves already executed by llm_ai_move()
                # NOTE: The LLM already executed the moves via make_move() calls,
                # which automatically recorded pairs in ai_pairs_accum via record_move_pair()
                if isinstance(llm_result, tuple):
                    # Multi-move: (token, num_moves_executed)
                    move_token, moves_executed = llm_result
                    
                    # Update dice state based on moves executed if needed
                    # Actually llm_ai_move updates the board and dice variables by calling make_move
                    # But we need to update moves_left_ai
                    moves_left_ai -= moves_executed
                    if moves_left_ai < 0: moves_left_ai = 0

                    print(f"✅ LLM turn complete: {moves_executed} moves executed")
                    
                    # If moves still left (e.g. partial sequence), fallback to search AI?
                    if moves_left_ai > 0 and move_token != '<NOMOVE>':
                         print(f"⚠️ LLM only made {moves_executed} moves, {moves_left_ai} remaining. Finishing with Search AI.")
                         while moves_left_ai > 0:
                            search_move_made = perform_search_ai_with_move_capture([dice1, dice2, dice3, dice4], moves_left_ai, player_turn)
                            if search_move_made:
                                moves_left_ai -= 1
                            else:
                                break
                else:
                    # Legacy single move token support
                    move_token = llm_result
                    moves_left_ai = 0  # Mark turn as complete
                    print(f"✅ LLM turn complete")
        
        # === SEARCH AI ONLY PATH ===
        else:
            # Search AI makes moves one at a time until no more possible
            moves_made = 0
            while moves_left_ai > 0:
                search_move_made = perform_search_ai_with_move_capture([dice1, dice2, dice3, dice4], moves_left_ai, player_turn)
                if search_move_made:
                    moves_left_ai -= 1
                    moves_made += 1
                else:
                    break

                # Add delay in versus mode so you can watch the AI vs AI games
                if versus_mode:
                    pygame.time.wait(500)  # 0.5 second delay between AI moves

            # If no moves were made by search_AI
            if moves_made == 0:
                print(f"No valid moves for AI with dice {dice1} {dice2}. Turn passes.")
                # Clear the message area first
                message_rect = pygame.Rect(10, HEIGHT+150, 500, 30)
                pygame.draw.rect(screen, WHITE_ISH, message_rect)
                text = font.render(f"AI has no valid moves with {dice1} {dice2}. Turn passes.", True, BLACK)
                screen.blit(text, (10, HEIGHT+150))

        # End of AI turn
        print(f"🤖 AI turn completed, {moves_left_ai} moves unused")
        
        # Flush history BEFORE switching player_turn (critical for versus mode!)
        flush_turn_history(player_turn)
        
        player_turn = -1 if player_turn == 1 else 1
        dice_rolled = False  # Reset dice for next player's turn

        dice3, dice4 = 0, 0
        rect = pygame.Rect(WIDTH//2, HEIGHT, WIDTH//2, 50)
        pygame.draw.rect(screen, WHITE_ISH, rect)
        display_turn(player_turn)
        dice_rolled = False
        if is_game_over():
            # Winner is whoever got 15 pieces off first
            if white_pieces_off == 15:
                winner = "Blue"
            elif black_pieces_off == 15:
                winner = "Red"
            else:
                winner = None  # Should not happen
            print(f"🎉 GAME OVER! {winner} wins!")
            print(f"🤖 LLM AI failures during this game: {llm_failures}")
            display_winner(winner)
            game_over = True

    draw_board(board, bar)
    draw_dice(dice1, dice2)
    pygame.display.flip()

    # Handle game over state
    if game_over:
        waiting_for_input = True
        while waiting_for_input and running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    waiting_for_input = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_r:
                        # Reset game
                        print("🔄 Restarting game...")
                        game_over = False
                        waiting_for_input = False

                        # Clear the bottom half of the screen
                        bottom_rect = pygame.Rect(0, HEIGHT, WIDTH, 200)
                        pygame.draw.rect(screen, WHITE_ISH, bottom_rect)

                        # Reset game state
                        board[:] = [0] * (BOARD_SIZE + 1)  # 25 elements (0-24), index 0 unused, positions 1-24
                        bar[:] = [0, 0]  # [white, black]
                        white_pieces_off = 0
                        black_pieces_off = 0
                        llm_failures = 0  # Reset LLM failure counter
                        # Standard backgammon starting positions (position = label)
                        # Blue pieces: start at 24, move counter-clockwise (24→1), bear off to 0
                        # Red pieces: start at 1, move clockwise (1→24), bear off to 25
                        board[24] = 2    # Blue: 2 pieces at position 24 (label 24, 'x')
                        board[13] = 5    # Blue: 5 pieces at position 13 (label 13, 'm')
                        board[8] = 3     # Blue: 3 pieces at position 8 (label 8, 'h')
                        board[6] = 5     # Blue: 5 pieces at position 6 (label 6, 'f')
                        board[1] = -2    # Red: 2 pieces at position 1 (label 1, 'a')
                        board[12] = -5   # Red: 5 pieces at position 12 (label 12, 'l')
                        board[17] = -3   # Red: 3 pieces at position 17 (label 17, 'q')
                        board[19] = -5   # Red: 5 pieces at position 19 (label 19, 's')
                        player_turn = 1
                        selected_piece = None
                        dice_rolled = False
                        dice1, dice2, dice3, dice4 = 0, 0, 0, 0
                        moves_left = 0
                        didnt_validate = False
                        game_history_tokens.clear()
                        game_history_tokens.append("<STARTGAME>")  # LLM needs this to start!
                        ai_pairs_accum.clear()
                        player_pairs_accum.clear()
                        current_player_dice = None
                        current_ai_dice = None
                        moves_left_ai = 0  # Reset AI moves counter
                        versus_mode = False  # Clear versus mode so it doesn't auto-start
                        initial_roll_done = False  # Reset initial roll state
                        player_initial_die = 0     # Reset initial die rolls
                        ai_initial_die = 0

                        # Reinitialize LLM predictor to clear any cached state
                        if llm_predictor is not None:
                            try:
                                llm_predictor = BackgammonMovePredictor()
                                print("✅ LLM predictor reinitialized after reset")
                            except Exception as e:
                                print(f"⚠️ Failed to reinitialize LLM predictor: {e}")
                                llm_predictor = None
                                ai_type = "search_AI"
                        print("👤🤖 HUMAN MODE: Reset to human vs AI mode")
                    elif event.key == pygame.K_q:
                        # Quit game
                        print("👋 Quitting game...")
                        running = False
                        waiting_for_input = False

pygame.quit()
