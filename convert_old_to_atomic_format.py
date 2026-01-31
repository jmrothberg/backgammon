"""
Convert old turn-based backgammon .txt format to new atomic tokenization format.

Old format: <STARTGAME> d41 m_lpab d66 m_xyz ... <EOFG>
New format: <STARTGAME> d4 d1 m_lp m_ab <EOM> d6 d6 m_xy m_z <EOM> ... <EOFG>

Conversion rules:
- Dice tokens: "d41" -> "d4", "d1" (split into individual dice)
- Move tokens: "m_lpab" -> "m_lp", "m_ab" (split into pairs)
- Add <EOM> after each turn
- Add <NOMOVE> if dice but no move
"""

import os
import tkinter as tk
from tkinter import filedialog
import re
from collections import Counter


def split_dice_token(dice_token):
    """
    Split old dice token into atomic dice tokens.
    
    Args:
        dice_token: Old format like "d41", "d66", "d55"
    
    Returns:
        List of atomic dice tokens: ["d4", "d1"] or ["d6", "d6"]
    """
    if not dice_token.startswith('d'):
        return []
    
    dice_str = dice_token[1:]  # Remove 'd' prefix
    
    if len(dice_str) == 2:
        die1 = dice_str[0]
        die2 = dice_str[1]
        return [f"d{die1}", f"d{die2}"]
    elif len(dice_str) == 1:
        # Single die (shouldn't happen in old format, but handle gracefully)
        return [f"d{dice_str}"]
    else:
        # Unexpected format
        return [dice_token]  # Keep original


def split_move_token(move_token):
    """
    Split old move token into atomic move pair tokens.
    
    Args:
        move_token: Old format like "m_lpab", "m_xyz", "m_a"
    
    Returns:
        List of atomic move tokens: ["m_lp", "m_ab"] or ["m_xy", "m_z"]
    """
    if not move_token.startswith('m_'):
        return []
    
    move_str = move_token[2:]  # Remove 'm_' prefix
    
    tokens = []
    for i in range(0, len(move_str), 2):
        if i + 1 < len(move_str):
            # Normal case: pair of characters
            pair = move_str[i:i+2]
            tokens.append(f"m_{pair}")
        else:
            # Edge case: odd length - single character
            tokens.append(f"m_{move_str[i]}")
    
    return tokens


def is_doubles(dice_token):
    """
    Check if dice token represents doubles (d11, d22, d33, d44, d55, d66).
    
    Args:
        dice_token: Dice token like "d66", "d41"
    
    Returns:
        True if doubles, False otherwise
    """
    if not dice_token.startswith('d'):
        return False
    
    dice_str = dice_token[1:]
    if len(dice_str) == 2 and dice_str[0] == dice_str[1]:
        return True
    
    return False


def convert_game_to_atomic(game_text):
    """
    Convert a single game from old format to new atomic format.
    
    Conversion rules:
    - Dice tokens: "d41" -> "d4", "d1" (split into individual dice)
    - Move tokens: "m_lpab" -> "m_lp", "m_ab" (split into pairs)
    - Doubles (d11-d66): Allow up to 4 move pairs
    - Add <EOM> after each turn (after moves or <NOMOVE>)
    - Add <NOMOVE> if dice but no moves
    
    Args:
        game_text: Old format game string
    
    Returns:
        New format game string with atomic tokens
    """
    # Split into tokens
    tokens = game_text.split()
    
    if not tokens:
        return None
    
    # Must start with <STARTGAME>
    if tokens[0] != '<STARTGAME>':
        return None
    
    # Must end with <EOFG>
    if tokens[-1] != '<EOFG>':
        return None
    
    # Process tokens (skip <STARTGAME> and <EOFG>)
    new_tokens = ['<STARTGAME>']
    i = 1  # Skip <STARTGAME>
    
    while i < len(tokens) - 1:  # Stop before <EOFG>
        token = tokens[i]
        
        # Check if it's a dice token
        if token.startswith('d') and len(token) > 1 and token[1:].isdigit():
            # Split dice into atomic tokens
            dice_atomic = split_dice_token(token)
            new_tokens.extend(dice_atomic)
            
            # Check if doubles (allows 4 moves)
            is_double = is_doubles(token)
            
            # Look ahead for move tokens
            moves_found = []
            j = i + 1
            while j < len(tokens) - 1 and tokens[j].startswith('m_'):
                moves_found.append(tokens[j])
                j += 1
            
            if moves_found:
                # Has moves - split all move tokens into atomic pairs
                for move_token in moves_found:
                    move_atomic = split_move_token(move_token)
                    new_tokens.extend(move_atomic)
                
                # Add <EOM> after all moves
                new_tokens.append('<EOM>')
                i = j  # Skip dice and all moves
            else:
                # No moves - add <NOMOVE> and <EOM>
                new_tokens.append('<NOMOVE>')
                new_tokens.append('<EOM>')
                i += 1  # Skip only dice
        
        # Check if it's a move token (without dice - shouldn't happen but handle)
        elif token.startswith('m_'):
            # Move without dice - split and add <EOM>
            move_atomic = split_move_token(token)
            new_tokens.extend(move_atomic)
            new_tokens.append('<EOM>')
            i += 1
        
        # Special tokens - keep as is (but <EOM> should already be added)
        elif token.startswith('<') and token.endswith('>'):
            # Skip if it's <STARTGAME> or <EOFG> (already handled)
            if token not in ['<STARTGAME>', '<EOFG>']:
                new_tokens.append(token)
            i += 1
        
        # Unknown format - skip
        else:
            i += 1
    
    # Add <EOFG>
    new_tokens.append('<EOFG>')
    
    return ' '.join(new_tokens)


def convert_file(old_file_path, output_file_path):
    """
    Convert entire file from old format to new atomic format.
    
    Args:
        old_file_path: Path to old format .txt file
        output_file_path: Path to save new format .txt file
    
    Returns:
        Tuple: (converted_text, statistics_dict)
    """
    # Read old format file
    with open(old_file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Split into games (games separated by blank lines)
    games = content.split('\n\n')
    games = [game.strip() for game in games if game.strip()]
    
    # Convert each game
    converted_games = []
    for game in games:
        converted = convert_game_to_atomic(game)
        if converted:
            converted_games.append(converted)
    
    # Combine converted games
    converted_text = '\n\n'.join(converted_games)
    
    # Calculate statistics
    all_tokens = converted_text.split()
    token_counts = Counter(all_tokens)
    unique_tokens = len(token_counts)
    total_tokens = len(all_tokens)
    
    # Most frequent tokens
    most_frequent = token_counts.most_common(10)
    
    # Rarest tokens (excluding special tokens for clarity)
    non_special_tokens = {k: v for k, v in token_counts.items() 
                         if not (k.startswith('<') and k.endswith('>'))}
    rarest = sorted(non_special_tokens.items(), key=lambda x: x[1])[:10]
    
    stats = {
        'total_games': len(converted_games),
        'total_tokens': total_tokens,
        'unique_tokens': unique_tokens,
        'most_frequent': most_frequent,
        'rarest': rarest,
        'token_counts': token_counts
    }
    
    # Write output file
    with open(output_file_path, 'w', encoding='utf-8') as f:
        f.write(converted_text)
    
    return converted_text, stats


def main():
    """
    Main conversion function with file selection and statistics display.
    """
    print("Backgammon Format Converter: Old Turn Format → Atomic Tokenization")
    print("=" * 70)
    
    # Select input file
    root = tk.Tk()
    root.withdraw()
    input_file = filedialog.askopenfilename(
        title="Select old format .txt file to convert",
        filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
    )
    
    if not input_file:
        print("No file selected. Exiting.")
        return
    
    print(f"📁 Input file: {input_file}")
    
    # Select output file
    output_file = filedialog.asksaveasfilename(
        title="Save converted atomic format file",
        defaultextension=".txt",
        initialfile="converted_atomic_format.txt",
        filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
    )
    
    if not output_file:
        print("No output file selected. Exiting.")
        return
    
    print(f"📁 Output file: {output_file}")
    print("\n🔄 Converting...")
    
    # Convert file
    try:
        converted_text, stats = convert_file(input_file, output_file)
        
        print("\n✅ Conversion complete!")
        print("=" * 70)
        
        # Display statistics
        print(f"\n📊 STATISTICS:")
        print(f"   🎮 Games converted: {stats['total_games']}")
        print(f"   🔢 Total tokens: {stats['total_tokens']:,}")
        print(f"   🎯 Unique tokens: {stats['unique_tokens']:,}")
        print(f"   📈 Token diversity: {stats['unique_tokens']/stats['total_tokens']:.4f}")
        
        # Most frequent tokens
        print(f"\n📈 TOP 10 MOST FREQUENT TOKENS:")
        for token, count in stats['most_frequent']:
            percentage = (count / stats['total_tokens']) * 100
            print(f"   '{token}': {count:,} times ({percentage:.2f}%)")
        
        # Rarest tokens
        print(f"\n📉 TOP 10 RAREST TOKENS:")
        for token, count in stats['rarest']:
            percentage = (count / stats['total_tokens']) * 100
            print(f"   '{token}': {count:,} times ({percentage:.4f}%)")
        
        # Show sample games
        games = converted_text.split('\n\n')
        games = [g.strip() for g in games if g.strip()]
        
        print(f"\n📋 SAMPLE CONVERTED GAMES:")
        print("=" * 70)
        
        # Show first game
        if len(games) > 0:
            print(f"\n🎮 Game 1 (first {min(300, len(games[0]))} characters):")
            print(games[0][:300] + "..." if len(games[0]) > 300 else games[0])
        
        # Show second game if available
        if len(games) > 1:
            print(f"\n🎮 Game 2 (first {min(300, len(games[1]))} characters):")
            print(games[1][:300] + "..." if len(games[1]) > 300 else games[1])
        
        # Token type breakdown
        dice_tokens = [t for t in stats['token_counts'] if t.startswith('d') and len(t) == 2 and t[1].isdigit()]
        move_tokens = [t for t in stats['token_counts'] if t.startswith('m_')]
        special_tokens = [t for t in stats['token_counts'] if t.startswith('<') and t.endswith('>')]
        
        print(f"\n🔢 TOKEN TYPE BREAKDOWN:")
        print(f"   🎲 Dice tokens (d1-d6): {len(dice_tokens)}")
        print(f"   🔄 Move tokens (m_xx): {len(move_tokens)}")
        print(f"   🎯 Special tokens: {len(special_tokens)}")
        
        print(f"\n✅ Conversion saved to: {output_file}")
        
    except Exception as e:
        print(f"\n❌ Error during conversion: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

