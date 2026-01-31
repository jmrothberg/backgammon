"""
BackgammonMovePredictor_Standalone.py

Standalone inference program for backgammon LLM models.
Handles device mapping issues automatically for robust inference across different hardware configurations.
Updated for Atomic Tokenization.
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torch.nn.functional as F
import tkinter as tk
from tkinter import filedialog
import re

# ============================================================================
# COPIED CLASSES AND FUNCTIONS FOR STANDALONE OPERATION
# ============================================================================

import math
from torch.optim.lr_scheduler import CosineAnnealingLR

# RMSNorm for stable training
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)
        x = x / rms
        return x * self.weight

# SwiGLU activation
class SwiGLU(nn.Module):
    def __init__(self, in_features, hidden_features=None):
        super().__init__()
        hidden_features = hidden_features or in_features * 4
        self.w1 = nn.Linear(in_features, hidden_features)
        self.w2 = nn.Linear(in_features, hidden_features)
        self.w3 = nn.Linear(hidden_features, in_features)

    def forward(self, x):
        gate = F.silu(self.w1(x))
        hidden = self.w2(x)
        return self.w3(gate * hidden)

# MultiQueryAttention (GQA)
class MultiQueryAttention(nn.Module):
    def __init__(self, n_embd, n_head, n_kv_heads, dropout):
        super().__init__()
        head_dim = n_embd // n_head
        self.n_heads = n_head
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim

        self.q_proj = nn.Linear(n_embd, n_head * head_dim)
        self.kv_proj = nn.Linear(n_embd, n_kv_heads * head_dim * 2)
        self.out_proj = nn.Linear(n_embd, n_embd)

        self.dropout = nn.Dropout(dropout)
        self.flash_available = hasattr(F, 'scaled_dot_product_attention')

    def forward(self, x, mask=None):
        B, T, C = x.size()

        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        kv = self.kv_proj(x).view(B, T, self.n_kv_heads, 2, self.head_dim)
        kv = kv.transpose(1, 2)
        k, v = kv[..., 0, :], kv[..., 1, :]

        k = k.repeat_interleave(self.n_heads // self.n_kv_heads, dim=1)
        v = v.repeat_interleave(self.n_heads // self.n_kv_heads, dim=1)

        if self.flash_available:
            causal_mask = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))
            if mask is not None:
                game_mask = mask[:, :T, :T].bool()
                combined_mask = torch.logical_and(causal_mask.unsqueeze(0), game_mask)
            else:
                combined_mask = causal_mask.unsqueeze(0)

            attention_mask = combined_mask.unsqueeze(1)

            try:
                y = F.scaled_dot_product_attention(
                    q, k, v,
                    attn_mask=attention_mask,
                    dropout_p=self.dropout.p if self.training else 0.0,
                    is_causal=False,
                    scale=1.0 / math.sqrt(k.size(-1)),
                    mem_efficient=True
                )
            except TypeError:
                y = F.scaled_dot_product_attention(
                    q, k, v,
                    attn_mask=attention_mask,
                    dropout_p=self.dropout.p if self.training else 0.0,
                    is_causal=False
                )
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            causal_mask = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))
            att = att.masked_fill(causal_mask == 0, float('-inf'))
            if mask is not None:
                att = att.masked_fill(mask[:, :T, :T].unsqueeze(1) == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.dropout(att)
            y = att @ v

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.out_proj(y)
        return y

# BackgammonBlock
class BackgammonBlock(nn.Module):
    def __init__(self, n_embd, n_head, n_kv_heads, dropout):
        super().__init__()
        self.rms_1 = RMSNorm(n_embd)
        self.attn = MultiQueryAttention(n_embd, n_head, n_kv_heads, dropout)
        self.rms_2 = RMSNorm(n_embd)
        self.swiglu = SwiGLU(n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        if self.training:
            x = x + self.dropout(torch.utils.checkpoint.checkpoint(
                lambda attn, rms_x, mask: attn(rms_x, mask=mask),
                self.attn, self.rms_1(x), mask, use_reentrant=False
            ))
            x = x + self.dropout(torch.utils.checkpoint.checkpoint(
                lambda ffwd, rms_x: ffwd(rms_x),
                self.swiglu, self.rms_2(x), use_reentrant=False
            ))
        else:
            x = x + self.dropout(self.attn(self.rms_1(x), mask=mask))
            x = x + self.dropout(self.swiglu(self.rms_2(x)))
        return x

# BackgammonModel
class BackgammonModel(nn.Module):
    def __init__(self, vocab_size, n_embd, n_head, n_kv_heads, block_size, n_layer, dropout, use_chess=False, use_dna=False):
        super().__init__()
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.use_chess = use_chess

        if use_chess:
            self.start_game_token = None

        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)

        self.blocks = nn.ModuleList([
            BackgammonBlock(n_embd, n_head, n_kv_heads, dropout)
            for _ in range(n_layer)
        ])

        self.rms_final = RMSNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def create_game_mask(self, idx):
        if not self.use_chess:
            return None
        mask = torch.ones_like(idx, dtype=torch.float32)
        game_boundaries = (idx == self.start_game_token).float().cumsum(dim=1)
        mask = (game_boundaries.unsqueeze(1) == game_boundaries.unsqueeze(2)).float()
        return mask

    def forward(self, idx, targets=None):
        B, T = idx.shape

        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))
        x = tok_emb + pos_emb

        for block in self.blocks:
            x = block(x, mask=self.create_game_mask(idx))

        x = self.rms_final(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits_flat = logits.view(B*T, C)
            targets_flat = targets.view(B*T)

            if hasattr(self, 'is_move_vec') and self.is_move_vec is not None:
                move_weights = getattr(self, 'move_weights', None)
                if move_weights is not None:
                    move_weights = move_weights.to(logits.device)
                    per_tok = F.cross_entropy(logits_flat, targets_flat, weight=move_weights, reduction='none')
                else:
                    per_tok = F.cross_entropy(logits_flat, targets_flat, reduction='none')

                with torch.no_grad():
                    mask = self.is_move_vec.to(targets_flat.device)[targets_flat]
                denom = torch.clamp(mask.sum(), min=1.0)
                loss = (per_tok * mask).sum() / denom
            else:
                loss = F.cross_entropy(logits_flat, targets_flat)

        return logits, loss

# Tokenization functions
def create_move_to_idx_from_text(text):
    token_to_idx = {}
    special_tokens = ['<STARTGAME>', '<EOFG>', '<EOM>', '<NOMOVE>', '<PAD>']

    for idx, token in enumerate(special_tokens):
        token_to_idx[token] = idx

    import re
    tokens = re.findall(r'(?:^| )([^ <][^ ]*?)(?= |$)', text)
    unique_tokens = set()
    for token in tokens:
        token = token.strip()
        if token and not token.startswith('<'):
            unique_tokens.add(token)

    sorted_tokens = sorted(unique_tokens)
    for token in sorted_tokens:
        if token not in token_to_idx:
            token_to_idx[token] = len(token_to_idx)

    print(f"Loaded tokenizer with {len(token_to_idx)} tokens")
    return token_to_idx

def create_idx_to_move(move_to_idx):
    return {idx: move for move, idx in move_to_idx.items()}

# Helper functions for atomic tokenization
def split_dice(dice_roll):
    """Split dice roll string into atomic dice tokens."""
    if len(dice_roll) == 2:
        return [f"d{dice_roll[0]}", f"d{dice_roll[1]}"]
    else:
        return [f"d{dice_roll}"]

def is_doubles(dice_roll):
    """Check if dice roll is doubles."""
    return len(dice_roll) == 2 and dice_roll[0] == dice_roll[1]

def get_top_k_pairs(model, dice_tokens, idx_to_move, move_to_idx, device, k=10, game_history=[]):
    """
    Get top-k move pairs with joint probabilities for non-doubles.
    """
    model.eval()
    
    # Build full context: game history + dice tokens
    input_tokens = game_history + dice_tokens
    context_indices = [move_to_idx[token] for token in input_tokens if token in move_to_idx]
    
    # Handle truncation if needed
    if len(context_indices) > model.block_size:
        context_indices = context_indices[-model.block_size:]
        
    context_tensor = torch.tensor([context_indices], dtype=torch.long).to(device)
    
    pairs = []
    
    with torch.no_grad():
        # Step 1: Get probability distribution for first move
        output, _ = model(context_tensor)
        first_logits = output[0, -1]  # Last position
        first_probs = torch.softmax(first_logits, dim=-1)
        
        # Get top candidates for first move
        # Look deeper to find enough valid move combinations
        top_first_k = min(50, len(first_probs))
        first_indices = torch.argsort(first_probs, descending=True)[:top_first_k]
        
        # Step 2: For each first move candidate, get second move probabilities
        for first_idx in first_indices:
            first_token = idx_to_move.get(first_idx.item(), None)
            
            # Check if it's a move token or NOMOVE
            if not first_token:
                continue
            
            # Allow NOMOVE
            if first_token == '<NOMOVE>':
                 first_prob = first_probs[first_idx].item()
                 pairs.append((first_prob, ['<NOMOVE>']))
                 continue
                 
            if not first_token.startswith('m_'):
                continue
            
            first_prob = first_probs[first_idx].item()
            
            # Extend context with first move
            # Be careful with block size here too
            extended_context = context_indices + [first_idx.item()]
            if len(extended_context) > model.block_size:
                extended_context = extended_context[-model.block_size:]
                
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
                if not second_token:
                    continue
                    
                # We expect a move or EOM
                if second_token == '<EOM>':
                    # Partial move sequence (1 move only)
                    second_prob = second_probs[second_idx].item()
                    joint_prob = first_prob * second_prob
                    pairs.append((joint_prob, [first_token]))
                    continue
                    
                if not second_token.startswith('m_'):
                    continue
                
                second_prob = second_probs[second_idx].item()
                
                # Joint probability: P(m1, m2) = P(m1) * P(m2 | m1)
                joint_prob = first_prob * second_prob
                
                pairs.append((joint_prob, [first_token, second_token]))
    
    # Step 4: Sort by joint probability and return top-k
    pairs.sort(key=lambda x: x[0], reverse=True)
    
    # Filter out partial moves that didn't end with EOM
    # Actually, for 2-move sequences, we might accept partial if probability is high enough
    # But ideally we want completed thoughts.
    return pairs[:k]

def get_top_k_quadruples(model, dice_tokens, idx_to_move, move_to_idx, device, k=10, game_history=[]):
    """
    Get top-k move quadruples with joint probabilities for doubles.
    Uses beam search to efficiently explore 4-move sequences.
    
    Handles:
    - <NOMOVE>: Returns immediately if predicted
    - <EOM>: Stops sequence generation early (e.g. for partial turns)
    - Max 4 moves: Stops after 4 moves
    """
    model.eval()
    
    # Build full context
    input_tokens = game_history + dice_tokens
    context_indices = [move_to_idx[token] for token in input_tokens if token in move_to_idx]
    
    if len(context_indices) > model.block_size:
        context_indices = context_indices[-model.block_size:]
        
    # Beam search: keep top-k sequences at each step
    # Increase beam width to avoid pruning valid but lower prob partial sequences early
    beam_width = k * 3
    
    # (context, prob, moves_list, finished)
    # finished=True means we hit <EOM> or <NOMOVE> or max length
    sequences = [(context_indices, 1.0, [], False)] 
    
    with torch.no_grad():
        # Doubles can have up to 4 moves
        for step in range(4): 
            candidates = []
            active_sequences = [s for s in sequences if not s[3]] # Only extend unfinished
            finished_sequences = [s for s in sequences if s[3]]   # Keep finished ones
            
            if not active_sequences:
                break
                
            for context, prob, moves, _ in active_sequences:
                # Truncate context if needed
                if len(context) > model.block_size:
                    ctx_input = context[-model.block_size:]
                else:
                    ctx_input = context
                    
                context_tensor = torch.tensor([ctx_input], dtype=torch.long).to(device)
                output, _ = model(context_tensor)
                logits = output[0, -1]
                probs = torch.softmax(logits, dim=-1)
                
                # Get top candidates for this step
                top_k_step = min(20, len(probs)) 
                top_indices = torch.argsort(probs, descending=True)[:top_k_step]
                
                for idx in top_indices:
                    token = idx_to_move.get(idx.item(), None)
                    if not token:
                        continue
                    
                    step_prob = probs[idx].item()
                    joint_prob = prob * step_prob
                    
                    # Handle termination conditions
                    if token == '<NOMOVE>':
                        if len(moves) == 0:
                            # Valid "No Move" prediction - mark as finished
                            candidates.append((context + [idx.item()], joint_prob, ['<NOMOVE>'], True))
                        continue
                        
                    if token == '<EOM>':
                         # Valid partial sequence end - mark as finished
                         # Don't include <EOM> in the moves list for the game engine, but stop generation
                         candidates.append((context + [idx.item()], joint_prob, moves, True))
                         continue
                         
                    if not token.startswith('m_'):
                        continue
                    
                    new_context = context + [idx.item()]
                    new_moves = moves + [token]
                    
                    # If we've reached 4 moves, this sequence is now finished
                    is_finished = (len(new_moves) == 4)
                    candidates.append((new_context, joint_prob, new_moves, is_finished))
            
            # Combine newly extended candidates with previously finished sequences
            all_pool = finished_sequences + candidates
            
            # Sort by probability and keep top beam_width
            all_pool.sort(key=lambda x: x[1], reverse=True)
            sequences = all_pool[:beam_width]
            
            # If all remaining sequences are finished, we can stop early
            if all(s[3] for s in sequences):
                break

    # Sort final results
    sequences.sort(key=lambda x: x[1], reverse=True)
    
    # Convert to expected return format (prob, moves)
    # Filter out empty moves unless it's explicitly NOMOVE
    results = []
    for _, prob, moves, _ in sequences:
        if moves: # Ensure we don't return empty lists
             results.append((prob, moves))
             
    return results[:k]

class BackgammonMovePredictor:
    """
    Standalone version with robust device handling for inference.
    Automatically maps models trained on any GPU configuration to available hardware.
    """

    def __init__(self, model_path=None, device=None):
        """
        Initialize with automatic device mapping.

        Args:
            model_path: Path to model checkpoint
            device: Target device (auto-detected if None)
        """
        # Auto-detect available device
        if device is None:
            if torch.cuda.is_available():
                device = torch.device('cuda:0')  # Always use GPU 0 for inference
            else:
                device = torch.device('cpu')

        self.device = device
        self.model = None
        self.idx_to_move = None
        self.move_to_idx = None

        if model_path:
            self.load_model(model_path)
        else:
            # Auto-detect model files in current directory
            import glob
            import os
            model_files = glob.glob("*.pth")
            if model_files:
                # Show available models and let user choose
                print("Available model files:")
                for i, model_file in enumerate(model_files, 1):
                    print(f"{i}. {model_file}")

                # Find the most recent one
                most_recent = max(model_files, key=os.path.getmtime)
                recent_idx = model_files.index(most_recent) + 1
                print(f"\nMost recent: {recent_idx}. {most_recent}")

                while True:
                    try:
                        choice = input(f"\nSelect model file (1-{len(model_files)}), or press Enter for most recent: ").strip()
                        if choice == "":
                            model_path = most_recent
                            print(f"Using most recent: {most_recent}")
                            break
                        choice_idx = int(choice) - 1
                        if 0 <= choice_idx < len(model_files):
                            model_path = model_files[choice_idx]
                            break
                        else:
                            print(f"Invalid choice. Please enter a number between 1 and {len(model_files)}.")
                    except ValueError:
                        print("Invalid input. Please enter a number or press Enter.")
                    except (EOFError, KeyboardInterrupt):
                        print("\nUsing most recent model.")
                        model_path = most_recent
                        break

                self.load_model(model_path)
            else:
                # Fallback to interactive loading if no models found
                print("No model files found in current directory.")
                print("Loading model interactively...")
                model_path = self._select_model_file()
                if model_path:
                    self.load_model(model_path)
                else:
                    raise ValueError("No model file selected")

    def _select_model_file(self):
        """Select model file with GUI or manual selection on Mac"""
        import platform
        if platform.system() == 'Darwin':  # macOS
            # List all .pth files for manual selection
            import glob
            model_files = glob.glob("*.pth")
            if not model_files:
                print("No .pth files found in current directory.")
                return None

            print("\nAvailable model files:")
            for i, model_file in enumerate(model_files, 1):
                print(f"{i}. {model_file}")

            while True:
                try:
                    choice = input(f"\nSelect model file (1-{len(model_files)}): ").strip()
                    choice_idx = int(choice) - 1
                    if 0 <= choice_idx < len(model_files):
                        return model_files[choice_idx]
                    else:
                        print(f"Invalid choice. Please enter a number between 1 and {len(model_files)}.")
                except ValueError:
                    print("Invalid input. Please enter a number.")
                except (EOFError, KeyboardInterrupt):
                    print("\nSelection cancelled.")
                    return None
        else:
            # Use GUI on other platforms
            root = tk.Tk()
            root.withdraw()
            file_path = filedialog.askopenfilename(
                title="Select Backgammon Model File",
                filetypes=[("PyTorch files", "*.pth")]
            )
            root.destroy()
            return file_path

    def load_model(self, model_path):
        """
        Load model with automatic device remapping.
        Handles models trained on any GPU configuration.
        """
        print(f"Loading model: {model_path}")

        # Create device map for any number of GPUs → available GPUs
        device_map = {}
        if torch.cuda.is_available():
            available_gpus = torch.cuda.device_count()
            # Map any training GPU (0,1,2,3,...) to available GPUs (cycle through available)
            for i in range(10):  # Support up to 10 GPUs in training
                device_map[f'cuda:{i}'] = f'cuda:{i % available_gpus}'
                device_map[i] = i % available_gpus  # Also handle int device indices

        # Load with device mapping
        try:
            checkpoint = torch.load(model_path, map_location=device_map)
        except Exception as e:
            print(f"Device mapping failed: {e}")
            print("Trying CPU fallback...")
            checkpoint = torch.load(model_path, map_location='cpu')
            self.device = torch.device('cpu')

        # Extract model configuration
        hyperparameters = checkpoint['hyperparameters']
        vocab_size = hyperparameters['vocab_size']
        n_embd = hyperparameters['n_embd']
        n_head = hyperparameters['n_head']
        n_kv_heads = hyperparameters.get('n_kv_heads', n_head // 4)
        block_size = hyperparameters['block_size']
        n_layer = hyperparameters['n_layer']
        dropout = hyperparameters['dropout']

        # Load tokenizer
        tokenizer = checkpoint.get('tokenizer')
        if isinstance(tokenizer, dict):
            self.move_to_idx = tokenizer
            self.idx_to_move = {idx: move for move, idx in self.move_to_idx.items()}
            print(f"Loaded tokenizer with {len(self.move_to_idx)} tokens")

        # Create model
        self.model = BackgammonModel(vocab_size, n_embd, n_head, n_kv_heads,
                                   block_size, n_layer, dropout, use_chess=True)

        # Load state dict with device compatibility
        state_dict = checkpoint['model_state_dict']

        # Clean state dict (remove module prefixes, skip problematic buffers)
        cleaned_state_dict = {}
        for key, val in state_dict.items():
            new_key = key
            if new_key.startswith('module.'):
                new_key = new_key[7:]
            elif new_key.startswith('_orig_mod.module.'):
                new_key = new_key[15:]
            elif new_key.startswith('_orig_mod.'):
                new_key = new_key[10:]

            # Skip buffers that will be recreated
            if new_key in ['is_move_vec']:
                continue

            cleaned_state_dict[new_key] = val

        # Load with strict=False to handle missing keys
        self.model.load_state_dict(cleaned_state_dict, strict=False)

        # Set start token
        self.model.start_game_token = self.move_to_idx.get('<STARTGAME>', 0)

        # Recreate move mask
        is_move_vec = torch.zeros(vocab_size, dtype=torch.float32)
        for tok, idx in self.move_to_idx.items():
            if isinstance(tok, str) and tok.startswith('m_'):
                is_move_vec[idx] = 1.0
        self.model.register_buffer('is_move_vec', is_move_vec)

        # Handle move_weights if present
        if 'move_weights' in cleaned_state_dict:
            move_weights_tensor = cleaned_state_dict['move_weights']
            self.model.register_buffer('move_weights', move_weights_tensor)
            print("✅ Loaded frequency-based weighting")
        else:
            self.model.move_weights = None
            print("📊 Using standard CrossEntropy loss")

        # Move model to device
        self.model = self.model.to(self.device)
        self.model.eval()

        print(f"✅ Model loaded successfully on {self.device}")
        print(f"   Model has {sum(p.numel() for p in self.model.parameters()):,} parameters")

    def predict_moves(self, game_history, dice, top_k=5):
        """
        Predict moves with automatic device handling.
        Returns atomic move sequences.

        Args:
            game_history: List of token strings
            dice: Dice roll string (e.g., "52")
            top_k: Number of predictions to return

        Returns:
            List of (confidence, move_sequence_list) tuples
        """
        if not self.model:
            raise ValueError("Model not loaded")

        return predict_backgammon_moves(
            self.model, game_history, dice,
            self.idx_to_move, self.move_to_idx,
            self.device, top_k=top_k
        )

# Updated predict function to use atomic tokenization logic
def predict_backgammon_moves(model, game_history, dice_roll, idx_to_move, move_to_idx, device, top_k=5):
    """
    Generate top-k move sequence predictions (pairs or quadruples).
    
    Returns:
        List of (joint_prob, move_sequence) tuples, sorted by probability
    """
    
    # Clean dice roll format if needed (e.g., d52 -> 52)
    if dice_roll.startswith('d') and len(dice_roll) > 2:
        clean_dice = dice_roll[1:]
    else:
        clean_dice = dice_roll

    # Split dice into atomic tokens
    dice_tokens = split_dice(clean_dice)  # "66" -> ["d6", "d6"]
    
    # Determine if doubles
    is_double = is_doubles(clean_dice)
    
    # Get predictions based on dice type
    if is_double:
        # Doubles: need up to 4 moves
        sequences = get_top_k_quadruples(
            model, dice_tokens, idx_to_move, move_to_idx, device, k=top_k, game_history=game_history
        )
    else:
        # Non-doubles: need up to 2 moves
        sequences = get_top_k_pairs(
            model, dice_tokens, idx_to_move, move_to_idx, device, k=top_k, game_history=game_history
        )
    
    return sequences


if __name__ == "__main__":
    print("🎮 Backgammon LLM Move Predictor (Standalone) - Atomic Tokenization")
    print("=" * 50)

    try:
        predictor = BackgammonMovePredictor()
        print("✅ Predictor ready for inference!")
        print("\nExample usage:")
        print("moves = predictor.predict_moves(game_history, '52', top_k=3)")
    except Exception as e:
        print(f"❌ Initialization failed: {e}")
        import traceback
        traceback.print_exc()
