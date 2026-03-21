"""
Backgammon SGF to TXT Converter

Converts a folder of SGF (Smart Game Format) backgammon files into a single
text file that can be tokenized and learned by an LLM for backgammon move prediction.

This program:
1. Reads all .sgf files from a selected folder
2. Parses each SGF file to extract backgammon moves
3. Converts moves to a text format suitable for LLM training
4. Outputs a single .txt file with all games

Format: <STARTGAME> d6 d1 m_ab m_cd <EOM> ... <EOFG>
"""

import os
import tkinter as tk
from tkinter import filedialog
import re
from datetime import datetime
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor


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


def parse_sgf_moves(sgf_content):
    """
    Parse SGF content to extract backgammon moves as ATOMIC tokens.

    Atomic Tokenization Strategy:
    - Dice: "d66" -> "d6", "d6" (split into individual dice)
    - Moves: "m_abcdefgh" -> "m_ab", "m_cd", "m_ef", "m_gh" (split into pairs)
    - Special: Adds <EOM> after each turn and <NOMOVE> if no moves made.

    SGF format for backgammon:
    ;B[move] - Black (first player) move
    ;W[move] - White (second player) move

    Args:
        sgf_content: Raw SGF file content as string

    Returns:
        List of atomic tokens
    """
    tokens = []

    # Find all move patterns: ;B[...] or ;W[...]
    # The moves are in brackets after B or W
    move_pattern = r';[BW]\[([^\]]+)\]'

    matches = re.findall(move_pattern, sgf_content)

    for match in matches:
        # Pure dice roll (no move made)
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
        
        # Move without explicit dice (shouldn't happen in standard SGF, but handle)
        elif re.match(r'^[a-zA-Z]+$', match):
            move_tokens = split_moves(match)
            tokens.extend(move_tokens)
            tokens.append('<EOM>')
        
        else:
            # Fallback for unexpected formats - keep original for debugging
            # tokens.append(match)
            pass

    return tokens


def process_sgf_file(file_path):
    """
    Process a single SGF file and return formatted game text with split tokens.

    Args:
        file_path: Path to the SGF file

    Returns:
        Formatted game string: "<STARTGAME> d3 d2 m_ad m_ln <EOM> ... <EOFG>"
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()

        # Parse moves from SGF as split tokens
        tokens = parse_sgf_moves(content)

        if not tokens:
            # print(f"Warning: No tokens found in {file_path}")
            return None

        # Format as space-separated tokens with game boundaries
        game_text = '<STARTGAME> ' + ' '.join(tokens) + ' <EOFG>'

        return game_text

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None


def convert_sgf_folder_to_txt():
    """
    Main function to convert a folder of SGF files to a single TXT file.
    """
    print("Backgammon SGF to TXT Converter (Atomic Tokenization)")
    print("=" * 50)

    # Select input folder
    root = tk.Tk()
    root.withdraw()
    folder_path = filedialog.askdirectory(
        title="Select folder containing SGF files"
    )

    if not folder_path:
        print("No folder selected. Exiting.")
        return

    print(f"Selected folder: {folder_path}")

    # Find all .sgf files recursively
    sgf_files = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith('.sgf'):
                sgf_files.append(os.path.join(root, file))

    if not sgf_files:
        print("No .sgf files found in the selected folder or subfolders.")
        return

    print(f"Found {len(sgf_files)} SGF files")

    # Process all SGF files concurrently for speed
    game_results = []
    processed_count = 0
    total_files = len(sgf_files)
    next_progress = 0
    progress_step = max(1, total_files // 20)  # Show progress ~20 times total

    print(f"Processing {total_files} files concurrently...")
    with ThreadPoolExecutor(max_workers=min(8, len(sgf_files))) as executor:
        # Submit all processing tasks
        future_to_file = {executor.submit(process_sgf_file, sgf_file): sgf_file for sgf_file in sorted(sgf_files)}

        # Collect results as they complete
        for future in concurrent.futures.as_completed(future_to_file):
            sgf_file = future_to_file[future]
            try:
                game_text = future.result()
                if game_text:
                    game_results.append((sgf_file, game_text))
                    processed_count += 1

                # Show progress at regular intervals
                if processed_count >= next_progress:
                    percent = (processed_count / total_files) * 100
                    print(f"Progress: {processed_count}/{total_files} files ({percent:.1f}%)")
                    next_progress += progress_step

            except Exception as e:
                print(f"✗ Error processing {os.path.basename(sgf_file)}: {e}")

    # Final progress update
    print(f"✓ Completed: {processed_count}/{total_files} files processed successfully")

    # Sort by filename for consistent ordering and extract game texts
    game_results.sort(key=lambda x: x[0])
    all_games = [game_text for _, game_text in game_results]

    if not all_games:
        print("No valid games could be processed.")
        return

    # Combine all games into single text
    final_text = '\n\n'.join(all_games)

    # Generate output filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"backgammon_atomic_games_{timestamp}.txt"

    # Select output location
    root = tk.Tk()
    root.withdraw()
    output_path = filedialog.asksaveasfilename(
        title="Save converted text file",
        defaultextension=".txt",
        initialfile=output_filename,
        filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
    )

    if not output_path:
        print("No output file selected. Exiting.")
        return

        # Write the combined text file
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(final_text)

        # Analyze token statistics
        all_tokens = final_text.split()
        unique_tokens = set(all_tokens)
        total_tokens = len(all_tokens)

        print("\n✅ Conversion complete!")
        print(f"📁 Output file: {output_path}")
        print(f"🎮 Games processed: {processed_count}")
        print(f"📝 Total characters: {len(final_text)}")
        print(f"🔢 Total tokens: {len(all_tokens):,}")

        # Calculate game lengths in tokens
        games = final_text.split('\n\n')
        games = [game.strip() for game in games if game.strip()]
        game_lengths_tokens = [len(game.split()) for game in games]
        avg_game_tokens = sum(game_lengths_tokens) // len(game_lengths_tokens)

        print(f"📊 Average game length: {avg_game_tokens} tokens")
        print(f"📏 Max game length: {max(game_lengths_tokens)} tokens")

        # Token analysis
        print(f"\n🔢 TOKEN ANALYSIS:")
        print(f"   📈 Total tokens: {total_tokens:,}")
        print(f"   🎯 Unique tokens: {len(unique_tokens):,}")
        print(f"   📊 Token diversity: {len(unique_tokens)/total_tokens:.4f} (unique/total)")

        # Analyze token patterns (split tokenization)
        dice_tokens = [t for t in unique_tokens if t.startswith('d') and t[1:].isdigit()]
        move_tokens = [t for t in unique_tokens if t.startswith('m_')]
        special_tokens = [t for t in unique_tokens if t.startswith('<') and t.endswith('>')]

        print(f"   🎲 Dice tokens (d1-d6): {len(dice_tokens):,}")
        print(f"   🔄 Move tokens (m_xx): {len(move_tokens):,}")
        print(f"   🎯 Special tokens: {len(special_tokens):,}")

        # Show sample of the output
        print("\n📋 Sample output format:")
        print(final_text.split('\n\n')[0][:200] + "...")

        # Show MOVE token distribution ONLY (ignore dice frequency)
        print("\n📈 Top 20 most frequent MOVE tokens (m_xx):")
        move_token_counts = {}
        for token in all_tokens:
            if token.startswith('m_'):  # Only count move tokens
                move_token_counts[token] = move_token_counts.get(token, 0) + 1

        sorted_move_tokens = sorted(move_token_counts.items(), key=lambda x: x[1], reverse=True)
        for token, count in sorted_move_tokens[:20]:
            print(f"   '{token}': {count:,} times")

    except Exception as e:
        print(f"Error writing output file: {e}")


if __name__ == "__main__":
    convert_sgf_folder_to_txt()
