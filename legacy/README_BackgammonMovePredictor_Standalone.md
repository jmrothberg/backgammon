# BackgammonMovePredictor_Standalone

A standalone inference program for backgammon LLM models that automatically handles device mapping issues when loading models trained on different GPU configurations.

## Problem Solved

When you train a backgammon model on multiple GPUs (e.g., 4 GPUs) and then try to use it for inference on a system with fewer GPUs (e.g., 1 GPU), PyTorch throws device mapping errors:

```
❌ Failed to initialize LLM predictor: Attempting to deserialize object on CUDA device 1 but torch.cuda.device_count() is 1. Please use torch.load with map_location to map your storages to an existing device.
```

## Solution

This standalone program automatically:
- Detects available GPUs on the inference system
- Maps any training GPU configuration to available hardware
- Falls back to CPU if GPU loading fails
- Handles all device compatibility issues transparently

## Features

- ✅ **Automatic device mapping** - Works with models trained on any GPU count
- ✅ **Robust error handling** - Graceful fallbacks for device issues
- ✅ **Standalone operation** - No dependencies on training code
- ✅ **Frequency weighting support** - Preserves weighting from training
- ✅ **Interactive model selection** - GUI file picker
- ✅ **Production ready** - Optimized for inference performance

## Usage

### Basic Usage
```python
from BackgammonMovePredictor_Standalone import BackgammonMovePredictor

# Initialize (auto-detects device and selects model)
predictor = BackgammonMovePredictor()

# Make predictions
game_history = ["<STARTGAME>", "d31", "m_adln", "d52", "m_mhxw"]
moves = predictor.predict_moves(game_history, "41", top_k=3)

for move, confidence in moves:
    print(f"{move}: {confidence:.1%}")
```

### Direct Model Loading
```python
# Load specific model
predictor = BackgammonMovePredictor(model_path="path/to/model.pth")

# Use on CPU if needed
predictor = BackgammonMovePredictor(model_path="path/to/model.pth", device="cpu")
```

## Device Mapping Logic

The program automatically handles these scenarios:

| Training System | Inference System | Action |
|----------------|------------------|---------|
| 4 GPUs (0,1,2,3) | 1 GPU (0) | Maps GPU 1→0, GPU 2→0, GPU 3→0 |
| 2 GPUs (0,1) | 4 GPUs (0,1,2,3) | Uses GPUs 0,1 as trained |
| Any GPU config | CPU only | Falls back to CPU |
| Multi-GPU | Same config | Uses original GPU mapping |

## Architecture

The standalone program includes:
- Complete model classes (BackgammonModel, MultiQueryAttention, etc.)
- Tokenization functions
- Device mapping utilities
- Error handling and fallbacks

## Benefits Over Training Code

1. **Lighter weight** - Only inference components, no training overhead
2. **Device robust** - Handles any hardware configuration mismatch
3. **Standalone** - No dependency on complex training setup
4. **Production ready** - Optimized for inference performance
5. **Maintainable** - Separate codebase for inference vs training

## Integration

Use this in your backgammon game engine:

```python
# In your game code
from BackgammonMovePredictor_Standalone import BackgammonMovePredictor

class BackgammonEngine:
    def __init__(self):
        self.predictor = BackgammonMovePredictor()

    def get_ai_move(self, game_state, dice):
        # Convert game state to token history
        history = self.game_state_to_tokens(game_state)

        # Get LLM predictions
        moves = self.predictor.predict_moves(history, dice, top_k=5)

        # Apply game rules and return best legal move
        return self.select_best_legal_move(moves, game_state)
```

## File Structure

```
BackgammonMovePredictor_Standalone.py
├── Model classes (BackgammonModel, MultiQueryAttention, etc.)
├── Device mapping utilities
├── Tokenization functions
├── Prediction logic
└── Error handling
```

This approach ensures reliable inference regardless of how or where the model was trained!
