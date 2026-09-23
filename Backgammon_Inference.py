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
    def __init__(self, vocab_size, n_embd, n_head, n_kv_heads, block_size, n_layer, dropout, use_chess=False, use_dna=False, use_value_head=False):
        super().__init__()
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.use_chess = use_chess
        self.use_value_head = use_value_head

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
        # Same head as the trainer: 0 = White won, 1 = Black won. Absent on older checkpoints.
        if self.use_value_head:
            self.head_value = nn.Linear(n_embd, 2)

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

    def _hidden(self, idx):
        """Hidden states at every position. Used by the win-guesser."""
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))
        x = tok_emb + pos_emb
        for block in self.blocks:
            x = block(x, mask=self.create_game_mask(idx))
        return self.rms_final(x)

    @torch.no_grad()
    def predict_value(self, idx):
        """Softmax over (White won, Black won) at the last position."""
        if not self.use_value_head:
            return None
        h = self._hidden(idx)
        return F.softmax(self.head_value(h[:, -1]), dim=-1)

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

def _fit_block(ids, block_size):
    """Keep the opening two tokens and the newest tail when the game is longer than the block.

    The opening two are <STARTGAME> and the result token (<W>, <B>, or <U>).
    Dropping them makes a long game forget which side is moving. The win-guesser uses this same cut.
    """
    if len(ids) <= block_size:
        return list(ids)
    head = 2 if block_size > 2 else 0
    return list(ids[:head]) + list(ids[-(block_size - head):])

def _batched_next_probs(model, contexts, device):
    """One forward for every partial turn. Logits are read at the last real token.

    Right-padding is safe because attention is causal: a pad after the real tokens
    cannot change the probability of the next move. Returns probs [batch, vocab].
    """
    block = model.block_size
    fitted = [_fit_block(c, block) for c in contexts]
    lengths = [max(len(c), 1) for c in fitted]
    max_t = max(lengths)
    batch = torch.zeros((len(fitted), max_t), dtype=torch.long, device=device)
    for i, c in enumerate(fitted):
        if c:
            batch[i, :len(c)] = torch.tensor(c, dtype=torch.long, device=device)
    logits, _ = model(batch)
    rows = torch.arange(len(fitted), device=device)
    cols = torch.tensor([n - 1 for n in lengths], device=device)
    return torch.softmax(logits[rows, cols], dim=-1)

def get_top_k_sequences(model, dice_tokens, idx_to_move, move_to_idx, device, k=10, game_history=None, max_moves=2, legal_fn=None):
    """Top-k finished turns.

    legal_fn(moves_so_far) lists the tokens that are legal on the board after those moves.
    When it is given, only those tokens are expanded, in one batched forward per checker.
    A turn is returned only when no checker can still be played (<EOM> or <NOMOVE>), or the
    dice are used up. A one-checker answer is not returned while a die can still move.

    Without legal_fn (no board), the model's own <EOM> / <NOMOVE> ends the turn.

    Also records the unmasked top token at the first checker, so play can still say when
    the model's own first choice was illegal.
    """
    if game_history is None:
        game_history = []
    model.eval()
    info = {'unmasked_first': None, 'unmasked_order': []}

    context_indices = [move_to_idx[token] for token in (list(game_history) + list(dice_tokens)) if token in move_to_idx]
    # (token ids, probability so far, move tokens)
    active = [(context_indices, 1.0, [])]
    finished = []
    # How many legal continuations to keep on each partial turn before the beam cut.
    # Keep enough legal first moves that the best finished turn is still in the beam.
    branch = 30
    beam_width = max(k * 4, branch)
    eom_id = move_to_idx.get('<EOM>')
    nomove_id = move_to_idx.get('<NOMOVE>')

    with torch.no_grad():
        for step in range(max_moves):
            if not active:
                break
            probs = _batched_next_probs(model, [seq[0] for seq in active], device)
            nxt = []
            for i, (context, prob, moves) in enumerate(active):
                step_probs = probs[i]
                if step == 0 and i == 0:
                    order_ids = torch.argsort(step_probs, descending=True)
                    order = []
                    for idx in order_ids.tolist():
                        token = idx_to_move.get(idx)
                        if token and (token.startswith('m_') or token in ('<NOMOVE>', '<EOM>')):
                            order.append(token)
                        if len(order) >= 30:
                            break
                    info['unmasked_order'] = order
                    info['unmasked_first'] = order[0] if order else idx_to_move.get(int(torch.argmax(step_probs).item()))

                allowed = None
                if legal_fn is not None:
                    allowed = set(legal_fn(moves))
                    move_tokens = [t for t in allowed if t.startswith('m_') and t in move_to_idx]
                    # No checker left: this partial turn is finished. Do not return it as a half turn.
                    if not move_tokens:
                        if '<NOMOVE>' in allowed and not moves and nomove_id is not None:
                            finished.append((prob * step_probs[nomove_id].item(), ['<NOMOVE>']))
                        elif eom_id is not None and moves:
                            finished.append((prob * step_probs[eom_id].item(), list(moves)))
                        continue
                    cand = [(step_probs[move_to_idx[t]].item(), move_to_idx[t], t) for t in move_tokens]
                else:
                    # No board here. Take the model's own move / stop tokens.
                    top_n = min(20, step_probs.shape[0])
                    cand = []
                    for idx in torch.argsort(step_probs, descending=True)[:top_n].tolist():
                        token = idx_to_move.get(idx)
                        if not token:
                            continue
                        if token == '<NOMOVE>' and not moves:
                            finished.append((prob * step_probs[idx].item(), ['<NOMOVE>']))
                            continue
                        if token == '<EOM>' and moves:
                            finished.append((prob * step_probs[idx].item(), list(moves)))
                            continue
                        if token.startswith('m_'):
                            cand.append((step_probs[idx].item(), idx, token))

                cand.sort(key=lambda item: item[0], reverse=True)
                for step_prob, idx, token in cand[:branch]:
                    nxt.append((context + [idx], prob * step_prob, moves + [token]))

            nxt.sort(key=lambda item: item[1], reverse=True)
            active = nxt[:beam_width]

        # Dice are used up. Keep the turn only when the board agrees nothing is left to play.
        # Multiply by P(<EOM>) so a short finished turn and a full turn are scored the same way.
        if active and legal_fn is not None:
            kept = []
            for context, prob, moves in active:
                allowed = set(legal_fn(moves))
                if not any(t.startswith('m_') for t in allowed):
                    kept.append((context, prob, moves))
            if kept and eom_id is not None:
                end_probs = _batched_next_probs(model, [seq[0] for seq in kept], device)
                for i, (context, prob, moves) in enumerate(kept):
                    finished.append((prob * end_probs[i, eom_id].item(), moves))
            else:
                finished.extend((prob, moves) for _context, prob, moves in kept)
        elif active and legal_fn is None:
            finished.extend((prob, moves) for _context, prob, moves in active if moves)

    finished.sort(key=lambda item: item[0], reverse=True)
    # Same moves reached by two paths: keep the higher probability.
    seen = set()
    results = []
    for prob, moves in finished:
        key = tuple(moves)
        if not moves or key in seen:
            continue
        seen.add(key)
        results.append((prob, moves))
        if len(results) >= k:
            break
    return results, info

def get_top_k_pairs(model, dice_tokens, idx_to_move, move_to_idx, device, k=10, game_history=[], legal_fn=None):
    """
    Get top-k move pairs with joint probabilities for non-doubles.
    Finished turns only: a one-checker answer is kept when the other die cannot be played.
    """
    results, _info = get_top_k_sequences(
        model, dice_tokens, idx_to_move, move_to_idx, device,
        k=k, game_history=game_history, max_moves=2, legal_fn=legal_fn,
    )
    return results

def get_top_k_quadruples(model, dice_tokens, idx_to_move, move_to_idx, device, k=10, game_history=[], legal_fn=None):
    """
    Get top-k move quadruples with joint probabilities for doubles.
    Uses beam search to efficiently explore 4-move sequences.

    Handles:
    - <NOMOVE>: Returns immediately if predicted
    - <EOM>: Stops sequence generation early (e.g. for partial turns)
    - Max 4 moves: Stops after 4 moves
    A stop before 4 moves is returned only when no checker can still be played.
    """
    results, _info = get_top_k_sequences(
        model, dice_tokens, idx_to_move, move_to_idx, device,
        k=k, game_history=game_history, max_moves=4, legal_fn=legal_fn,
    )
    return results

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

        # Older checkpoints have no win head. Do not invent one — play then keeps the first legal turn.
        has_value_head = any(
            key == 'head_value.weight' or key.endswith('head_value.weight')
            for key in cleaned_state_dict
        )

        # Create model
        self.model = BackgammonModel(vocab_size, n_embd, n_head, n_kv_heads,
                                   block_size, n_layer, dropout, use_chess=True,
                                   use_value_head=has_value_head)

        # Load with strict=False to handle missing keys
        self.model.load_state_dict(cleaned_state_dict, strict=False)
        self.model._has_value_head = bool(has_value_head and self.model.use_value_head)
        if self.model._has_value_head:
            print("Win-guesser loaded (value head). A later legal turn can replace the 1st pick.")
        else:
            print("No win-guesser on this checkpoint. The first legal turn is played.")

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

    def with_result_token(self, game_history, result_token):
        """Put <W>, <B>, or <U> immediately after <STARTGAME>.

        Older checkpoints have none of those tokens, so the history is returned unchanged
        and play stays exactly as it was.
        """
        if not self.move_to_idx or '<U>' not in self.move_to_idx or '<W>' not in self.move_to_idx:
            return list(game_history)
        hist = list(game_history)
        if not hist:
            return ['<STARTGAME>', result_token]
        if hist[0] != '<STARTGAME>':
            return hist
        if len(hist) > 1 and hist[1] in ('<W>', '<B>', '<U>'):
            hist[1] = result_token
        else:
            hist.insert(1, result_token)
        return hist

    @torch.no_grad()
    def rerank_by_value(self, game_history, dice, side, sequences):
        """Score each full turn with the win head.

        Prompt is <STARTGAME> <U> history dice moves, so the head cannot read a result token.
        Score is P(our color wins) - P(our color loses). side is 'W' or 'B'.
        sequences is a list of (probability, token list).
        Returns {tuple(tokens): score}. Empty when this checkpoint has no value head.
        """
        if not getattr(self.model, '_has_value_head', False) or not sequences:
            return {}
        if '<U>' not in self.move_to_idx:
            return {}
        base = self.with_result_token(game_history, '<U>')
        clean_dice = dice[1:] if isinstance(dice, str) and dice.startswith('d') and len(dice) > 2 else dice
        prefix = base + split_dice(clean_dice)
        block = self.model.block_size
        scores = {}
        for _prob, seq in sequences:
            moves = [tok for tok in seq if tok and tok != '<EOM>']
            ids = [self.move_to_idx[tok] for tok in (prefix + moves) if tok in self.move_to_idx]
            if not ids:
                continue
            # Keep <STARTGAME> <U> and the newest tokens when the game is longer than the block.
            ids = _fit_block(ids, block)
            idx = torch.tensor([ids], dtype=torch.long, device=self.device)
            probs = self.model.predict_value(idx)
            if probs is None:
                return {}
            white_win = probs[0, 0].item()
            black_win = probs[0, 1].item()
            score = (white_win - black_win) if side == 'W' else (black_win - white_win)
            scores[tuple(seq)] = score
        return scores

    def predict_moves(self, game_history, dice, top_k=5, legal_fn=None):
        """
        Predict moves with automatic device handling.
        Returns atomic move sequences.

        Args:
            game_history: List of token strings
            dice: Dice roll string (e.g., "52")
            top_k: Number of predictions to return
            legal_fn: optional legal_fn(moves_so_far) -> tokens that are legal now.
                When set, every returned turn is finished and legal.

        Returns:
            List of (confidence, move_sequence_list) tuples
        """
        if not self.model:
            raise ValueError("Model not loaded")

        meta = {}
        sequences = predict_backgammon_moves(
            self.model, game_history, dice,
            self.idx_to_move, self.move_to_idx,
            self.device, top_k=top_k, legal_fn=legal_fn, meta=meta,
        )
        # Unmasked top token, before the legal mask, so play can report a bad first choice.
        self.last_unmasked_first = meta.get('unmasked_first')
        self.last_unmasked_order = meta.get('unmasked_order') or []
        return sequences

# Updated predict function to use atomic tokenization logic
def predict_backgammon_moves(model, game_history, dice_roll, idx_to_move, move_to_idx, device, top_k=5, legal_fn=None, meta=None):
    """
    Generate top-k move sequence predictions (pairs or quadruples).

    Returns:
        List of (joint_prob, move_sequence) tuples, sorted by probability.
        With legal_fn, each sequence is a finished legal turn.
    """

    # Clean dice roll format if needed (e.g., d52 -> 52)
    if dice_roll.startswith('d') and len(dice_roll) > 2:
        clean_dice = dice_roll[1:]
    else:
        clean_dice = dice_roll

    # Split dice into atomic tokens
    dice_tokens = split_dice(clean_dice)  # "66" -> ["d6", "d6"]

    # Doubles can use four checkers. Any other roll uses two.
    max_moves = 4 if is_doubles(clean_dice) else 2
    sequences, info = get_top_k_sequences(
        model, dice_tokens, idx_to_move, move_to_idx, device,
        k=top_k, game_history=game_history, max_moves=max_moves, legal_fn=legal_fn,
    )
    if meta is not None:
        meta.update(info)
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
