#!/usr/bin/env python3
"""
Generate full backgammon games using GNU Backgammon
"""
import subprocess
import time
import os
import sys
from datetime import datetime
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

def run_gnubg_command(commands, timeout=300):
    """Run gnubg with commands and return output"""
    try:
        cmd = ['gnubg', '-t', '-q']
        process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env={**os.environ, 'DISPLAY': '', 'GNUBG_GUI': '0'}
        )

        # Send commands
        command_str = '\n'.join(commands) + '\n'
        stdout, stderr = process.communicate(input=command_str, timeout=timeout)

        return process.returncode == 0, stdout, stderr
    except subprocess.TimeoutExpired:
        process.kill()
        return False, "", "Timeout"
    except Exception as e:
        return False, "", str(e)

def generate_single_game(game_num, output_dir, instance_id=0):
    """Generate a single backgammon game using GNU Backgammon"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create unique seed for reproducible but varied games:
    # - game_num: differentiates games within an instance
    # - instance_id * 1000000: separates different parallel instances (prevents overlap)
    # - base_seed from timestamp: ensures different runs produce different games
    base_seed = int(timestamp.replace('_', ''))  # Convert timestamp to number
    unique_game_id = game_num + (instance_id * 1000000) + base_seed

    # GNU Backgammon commands for headless game generation
    commands = [
        'set automatic game on',        # Auto-start new games
        'set automatic roll on',        # Auto-roll dice
        'set automatic move on',        # Auto-make moves
        'set player 0 gnubg ply 2',     # Player 1: Intermediate level (2-ply search depth)
        'set player 1 gnubg ply 2',     # Player 2: Intermediate level (2-ply search depth)
        'set matchlength 1',            # Single game matches (not multi-game matches)
        'set rng isaac',                # Use Isaac RNG (deterministic but high quality)
        f'set seed {unique_game_id}',   # Set seed for reproducible games
        'new match',                    # Start new match
        'play',                         # Play the game
        f'save match "{output_dir}/game_{timestamp}_{instance_id}_{game_num}.sgf"',  # Save in SGF format
        'quit'                          # Exit gnubg
    ]

    success, stdout, stderr = run_gnubg_command(commands)

    filename = f"{output_dir}/game_{timestamp}_{instance_id}_{game_num}.sgf"
    if os.path.exists(filename):
        # Count moves
        with open(filename, 'r') as f:
            content = f.read()
        move_count = content.count(';B[') + content.count(';W[')
        return True, move_count
    else:
        return False, 0

def generate_games_parallel(num_games, output_dir, instance_id=0, max_workers=None):
    """Generate games in parallel"""
    if max_workers is None:
        max_workers = min(32, mp.cpu_count())  # Default to 32 cores max

    print(f"🚀 Starting parallel generation with {max_workers} workers (instance {instance_id})")

    success_count = 0
    total_moves = 0
    completed_games = 0
    report_interval = 100  # Report every 100 games

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_game = {executor.submit(generate_single_game, i, output_dir, instance_id): i
                         for i in range(1, num_games + 1)}

        # Process results as they complete
        for future in as_completed(future_to_game):
            game_num = future_to_game[future]
            try:
                success, moves = future.result()
                completed_games += 1

                if success:
                    success_count += 1
                total_moves += moves  # Count moves regardless of success

                # Report progress periodically
                if completed_games % report_interval == 0 or completed_games == num_games:
                    progress = (completed_games / num_games) * 100
                    print(f"📊 Instance {instance_id}: {completed_games}/{num_games} games ({progress:.1f}%) - {success_count} successful")

            except Exception as exc:
                completed_games += 1
                print(f"❌ Instance {instance_id} Game {game_num} exception: {exc}")

    return success_count, total_moves

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python generate_backgammon_games.py <num_games> <output_dir> [instance_id] [max_workers]")
        sys.exit(1)

    num_games = int(sys.argv[1])
    output_dir = sys.argv[2]
    instance_id = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    max_workers = int(sys.argv[4]) if len(sys.argv) > 4 else None

    if num_games <= 0:
        print("Error: Number of games must be positive")
        sys.exit(1)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Use parallel generation
    success_count, total_moves = generate_games_parallel(num_games, output_dir, instance_id, max_workers)

    print(f"\n🎉 Generation complete!")
    print(f"✅ Successfully generated: {success_count}/{num_games} games")
    if success_count > 0:
        print(f"📊 Average moves per game: {total_moves / success_count:.1f}")
    print(f"📁 Games saved in: {output_dir}/")
    print(f"🏷️  Instance ID: {instance_id}")
