#!/bin/bash

# Backgammon Game Generation Setup for macOS
# This script installs GNU Backgammon and generates backgammon games for LLM training
# Runs completely headless without GUI
#
# Usage: ./setup_backgammon_games.sh [number_of_games]
# Examples:
#   ./setup_backgammon_games.sh        # Generate 100 games (default)
#   ./setup_backgammon_games.sh 500    # Generate 500 games
#   ./setup_backgammon_games.sh 1000   # Generate 1000 games

set -e  # Exit on any error

# Parse command line arguments
NUM_GAMES=${1:-100}  # Default to 100 games if no argument provided

# Validate input
if ! [[ "$NUM_GAMES" =~ ^[0-9]+$ ]] || [ "$NUM_GAMES" -le 0 ]; then
    echo "❌ Error: Please provide a positive integer for number of games"
    echo "Usage: $0 [number_of_games]"
    echo "Examples:"
    echo "  $0           # Generate 100 games (default)"
    echo "  $0 500       # Generate 500 games"
    echo "  $0 1000      # Generate 1000 games"
    exit 1
fi

echo "🎲 Backgammon Game Generation Setup for macOS"
echo "=============================================="
echo "📊 Generating $NUM_GAMES games"

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to install Homebrew if not present
install_homebrew() {
    echo "📦 Installing Homebrew..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

    # Add Homebrew to PATH for this session
    if [[ -f /opt/homebrew/bin/brew ]]; then
        eval "$(/opt/homebrew/bin/brew shellenv)"
    elif [[ -f /usr/local/bin/brew ]]; then
        eval "$(/usr/local/bin/brew shellenv)"
    else
        echo "❌ Error: Could not find brew after installation"
        exit 1
    fi
}

# Function to install GNU Backgammon
install_gnubg() {
    echo "🎯 Installing GNU Backgammon..."

    # Update Homebrew
    brew update

    # Install GNU Backgammon
    if brew install gnubg; then
        echo "✅ GNU Backgammon installed successfully"
    else
        echo "❌ Failed to install GNU Backgammon"
        echo "Trying alternative installation method..."

        # Alternative: build from source
        echo "🔧 Building GNU Backgammon from source..."
        cd /tmp

        # Clean up any existing gnubg directory
        rm -rf gnubg

        # Install build dependencies
        echo "📦 Installing build dependencies..."
        brew install autoconf automake libtool pkg-config
        brew install gtk+3 cairo librsvg libxml2 gettext

        # Use the correct GNU Savannah repository
        git clone https://git.savannah.gnu.org/git/gnubg.git
        cd gnubg

        # Generate build files
        ./autogen.sh

        # Configure with GUI disabled for headless operation
        ./configure --disable-gui --without-gtk --without-board3d \
                   --prefix=/usr/local

        # Build and install
        make -j$(sysctl -n hw.ncpu)
        make install
        cd -
        rm -rf /tmp/gnubg
    fi
}

# Function to verify installation
verify_installation() {
    echo "🔍 Verifying GNU Backgammon installation..."

    if command_exists gnubg; then
        echo "✅ GNU Backgammon found at: $(which gnubg)"
        gnubg --version
        return 0
    else
        echo "❌ GNU Backgammon not found after installation"
        return 1
    fi
}

# Function to generate games
generate_games() {
    echo "🎮 Generating backgammon games using Python script..."

    # Create games directory
    mkdir -p backgammon_games

    # Run the Python script to generate games
    echo "Generating $NUM_GAMES full backgammon games with 2-ply analysis..."
    python3 generate_backgammon_games.py "$NUM_GAMES" "backgammon_games"

    echo "🎉 Game generation complete!"
    GAME_COUNT=$(ls backgammon_games/game_*.sgf 2>/dev/null | wc -l | tr -d ' ')
    echo "📊 Total games generated: $GAME_COUNT"

    # Analyze the generated games
    if [ "$GAME_COUNT" -gt 0 ]; then
        echo "📈 Game analysis:"
        ls backgammon_games/game_*.sgf | head -5 | while read file; do
            MOVE_COUNT=$(grep -c ";[BW]\[" "$file" 2>/dev/null || echo "0")
            echo "  $(basename "$file"): $MOVE_COUNT moves"
        done
    fi

    echo "📁 Games saved in: $(pwd)/backgammon_games/"
}

# Main execution
main() {
    echo "🚀 Starting Backgammon setup and game generation..."

    # Check operating system
    if [[ "$OSTYPE" == "darwin"* ]]; then
        echo "🍎 Running on macOS"
        PACKAGE_MANAGER="brew"
    elif [[ -f /etc/debian_version ]]; then
        echo "🐧 Running on Debian/Ubuntu Linux"
        PACKAGE_MANAGER="apt"
    else
        echo "❓ Unknown operating system: $OSTYPE"
        echo "This script supports macOS and Debian/Ubuntu Linux"
        exit 1
    fi

    # Check/install GNU Backgammon based on OS
    if [[ "$PACKAGE_MANAGER" == "brew" ]]; then
        # macOS with Homebrew
        if ! command_exists brew; then
            echo "🍺 Homebrew not found. Installing..."
            install_homebrew
        else
            echo "✅ Homebrew found at: $(which brew)"
        fi

        if ! command_exists gnubg; then
            install_gnubg
        else
            echo "✅ GNU Backgammon already installed"
        fi
    elif [[ "$PACKAGE_MANAGER" == "apt" ]]; then
        # Linux with apt
        if ! command_exists gnubg; then
            echo "📦 Installing GNU Backgammon via apt..."
            sudo apt-get update
            sudo apt-get install -y gnubg
        else
            echo "✅ GNU Backgammon already installed"
        fi
    fi

    # Verify installation
    if ! verify_installation; then
        echo "❌ Installation verification failed"
        exit 1
    fi

    # Generate games
    generate_games

    echo ""
    echo "🎯 Setup Complete!"
    echo "=================="
    echo "✅ GNU Backgammon installed and working"
    echo "✅ $NUM_GAMES backgammon games generated"
    echo "✅ Games saved with timestamps (no overwrites)"
    echo "✅ Ready for LLM training"
    echo ""
    echo "📝 Next steps:"
    echo "1. Parse the .sgf files to extract move sequences"
    echo "2. Convert moves to format: <STARTGAME> 24/18 13/9 ... <EOFG>"
    echo "3. Create backgammon vocabulary for your LLM"
    echo "4. Train your BackgammonBrain model!"
}

# Run main function
main "$@"
