"""
BackgammonBrain - Complete Neural Network Backgammon AI System

Full implementation for training and deploying transformer-based backgammon AI.

ARCHITECTURE OVERVIEW:
- Specialized transformer for backgammon strategy learning
- Split tokenization: dice tokens (d11-d66) + move tokens (m_xxxx)
- MultiQueryAttention for memory-efficient attention computation
- RMSNorm for stable training, SwiGLU for better activation
- Game boundary masking prevents attention across different games

TRAINING APPROACH:
- Learns from expert GNU Backgammon games in SGF format
- Autoregressive prediction: next token given full game context
- Statistical learning of winning patterns across game phases
- Strategic understanding: bearing off, racing, blocking, hitting

INFERENCE CAPABILITIES:
- Top-k move predictions with confidence scores
- Full game history context (like chess PGN tracking)
- Legal move filtering and fallback handling
- Production API via BackgammonMovePredictor class

INTEGRATION FEATURES:
- Clean game engine interface with error handling
- Console logging for debugging LLM decisions
- Automatic fallback from neural network to search AI
- CRITICAL: Maintains complete game history as token sequences
- Full context tracking ensures LLM strategic understanding

USAGE WORKFLOW:
1. Train: Convert SGF games → train transformer → save model
2. Deploy: BackgammonMovePredictor(model) → predict_moves() → game integration
3. Play: LLM provides strategic suggestions, search AI ensures legality
"""

import re
import os
import random   # <U> result masking in BackgammonMovesDataset

# PyTorch 2.9+ on this GB10 ignores PYTORCH_CUDA_ALLOC_CONF. The new name must be
# set before torch is imported or the allocator never sees it (fragmented unified memory).
if not os.environ.get('PYTORCH_ALLOC_CONF'):
    os.environ['PYTORCH_ALLOC_CONF'] = 'max_split_size_mb:512,expandable_segments:True,garbage_collection_threshold:0.8'
# Same as Chess: Triton's own ptxas does not know this Spark's sm_121a, so torch.compile
# needs the system CUDA assembler before torch is imported.
if not os.environ.get('TRITON_PTXAS_PATH'):
    _cuda13_ptxas = '/usr/local/cuda/bin/ptxas'
    if os.path.isfile(_cuda13_ptxas):
        os.environ['TRITON_PTXAS_PATH'] = _cuda13_ptxas

# PLAIN LANGUAGE: Debug switches for inference (safe to delete later)
# - DEBUG_PREDICTIONS: turn on/off one-shot debug prints during prediction
# - PRED_DEBUG_LIMIT: only print this many times per run (prevents spam)
# - These do NOT change model behavior. They only print a short snapshot
#   so you (and future LLMs) can see: raw top tokens, which are moves vs dice,
#   and how much probability is on moves overall.
DEBUG_PREDICTIONS = True
PRED_DEBUG_LIMIT = 5
_pred_debug_calls = 0
_vocab_dupe_checked = False
import platform
import torch
import torch.nn as nn
import torch.optim as optim
import math
from datetime import datetime
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import tkinter as tk
from tkinter import filedialog
from torch.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from transformers.optimization import Adafactor

def _uses_unified_memory():
    """True when CUDA memory is the same RAM the desktop uses (NVIDIA GB10 / DGX Spark)."""
    if not torch.cuda.is_available():
        return False
    try:
        if 'GB10' in torch.cuda.get_device_name(0).upper():
            return True
        gpu = float(torch.cuda.get_device_properties(0).total_memory)
        sys_ram = float(os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES'))
        return abs(gpu - sys_ram) / max(sys_ram, 1.0) < 0.15
    except Exception:
        return False


# Leave this much RAM free on the Spark. A fixed 80% of the whole 128 GB pool freezes the desktop.
UNIFIED_HEADROOM_BYTES = 16 * (1024 ** 3)


def _unified_cuda_fraction(free, total):
    """CUDA cap = memory that is free now, minus headroom for the desktop."""
    usable = max(UNIFIED_HEADROOM_BYTES, free - UNIFIED_HEADROOM_BYTES)
    return max(0.15, min(0.70, usable / max(total, 1)))


# Prioritize MPS on Mac systems for native GPU support
if platform.system() == "Darwin" and torch.backends.mps.is_available():
    device = torch.device('mps')
    gpu_indices = []  # MPS doesn't use gpu_indices like CUDA
    print("Using MPS GPU")
elif torch.cuda.is_available():
    # GPU selection will be done later - either from checkpoint or user selection
    # For now, just initialize to None and set device
    gpu_indices = None

    # PYTORCH_ALLOC_CONF is set above, before torch is imported.

    # Train faster by allowing TF32 precision on A100 and newer GPUs if available
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    torch.backends.cudnn.benchmark = True

    # Reduce memory fragmentation
    torch.cuda.empty_cache()

    # Better asynchronous GPU operations
    torch.cuda.set_stream(torch.cuda.Stream())

    # Force garbage collection to reduce memory fragmentation
    import gc
    gc.collect()

    device = torch.device('cuda')
    print("Using CUDA with optimized settings")
    # Set CUDA to release memory when possible - helps prevent OOM errors
    torch.cuda.empty_cache()
    # Discrete GPUs can take 80% of VRAM. On the Spark, CUDA and the desktop share one pool,
    # so the cap is "what is free now, minus 16 GB", the same rule as Chess.
    if _uses_unified_memory():
        _free, _total = torch.cuda.mem_get_info(0)
        _frac = _unified_cuda_fraction(_free, _total)
        torch.cuda.set_per_process_memory_fraction(_frac)
        print(f"GB10 unified memory: CUDA capped at {_frac:.0%} of RAM "
              f"(keeps {UNIFIED_HEADROOM_BYTES/2**30:.0f} GB free for the desktop)")
    else:
        torch.cuda.set_per_process_memory_fraction(0.80)
else:
    print("❌ ERROR: No GPU available. This backgammon training requires GPU support (CUDA or MPS).")
    exit(1)

# At the beginning of your script, after device selection:
if device.type == 'mps':
    torch.set_default_dtype(torch.float32)

# Define special tokens for backgammon games
special_tokens = ['<STARTGAME>', '<EOFG>']

# Backgammon defaults - optimized for split tokenization and smaller vocabulary
BACKGAMMON_DEFAULTS = {
    # PLAIN: Bigger model so it can learn rules and 2-3 ply patterns
    'n_embd': 384,       # was 128; 384 gives better capacity for legality patterns
    'n_head': 8,         # keep heads the same (works well)
    'n_kv_heads': 2,     # GQA ratio (efficient)
    'block_size': 512,   # Increased for atomic tokens (longer sequences)
    'n_layer': 8,       # was 8; deeper helps with bar/bear-off/doubles logic
    'dropout': 0.1,      # safe regularization
    'batch_size': 128,   # keep stable; raise only if GPU allows
    'num_epochs': 5,     # session-oriented training
    'learning_rate': 4e-4,  # stable LR for this size
    'weight_decay': 0.01,   # standard
    'max_norm': 5.0,     # gradient clipping
    # Same idea as Chess (Sep 2026): loser moves count less, and a small head guesses who won.
    'loser_move_weight': 0.5,  # weight on checker moves (m_*) played by the side that lost
    'value_loss_weight': 0.2,  # weight of the auxiliary White/Black win head
    'value_mask_prob': 0.5,    # fraction of games whose input result token is replaced by <U>
}

# Core model components for backgammon move prediction
class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.

    RMSNorm provides an efficient alternative to standard LayerNorm by normalizing
    only by the root mean square of the features, without centering (mean subtraction).
    This reduces computation while maintaining training stability.

    Key advantages over LayerNorm:
    - Faster computation (no mean calculation)
    - Better gradient flow in deep networks
    - Equivalent performance to LayerNorm in practice

    Args:
        dim: Feature dimension to normalize
        eps: Small epsilon for numerical stability (default: 1e-5)
    """
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # Compute RMS normalization: x / sqrt(mean(x^2) + eps)
        rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)
        x = x / rms
        return x * self.weight


class FeedForward(nn.Module):
    """
    Feed-forward network with expansion and contraction layers.

    Standard transformer feed-forward network that expands input dimension by 4x,
    applies non-linearity, then contracts back to original dimension. Used in
    transformer blocks after attention layers.

    Architecture:
    - Linear expansion: n_embd → 4*n_embd
    - ReLU activation for non-linearity
    - Linear contraction: 4*n_embd → n_embd
    - Dropout for regularization

    Args:
        n_embd: Input/output embedding dimension
        dropout: Dropout probability applied after final linear layer
    """
    def __init__(self, n_embd, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class MultiHeadAttention(nn.Module):
    """
    Multi-head self-attention mechanism with optional RoPE positional embeddings.

    Implements scaled dot-product attention with multiple attention heads.
    Supports both Flash Attention (PyTorch 2.0+) and traditional attention implementations.
    Can use Rotary Position Embeddings (RoPE) for better sequence understanding,
    particularly effective for longer sequences.

    Key features:
    - Multi-head attention for capturing different attention patterns
    - Causal masking for autoregressive generation
    - Optional RoPE for position-aware attention
    - Automatic fallback between Flash Attention and traditional implementation

    Args:
        n_embd: Embedding dimension (must be divisible by n_head)
        n_head: Number of attention heads
        block_size: Maximum sequence length for masking
        dropout: Dropout probability applied to attention weights
        use_rope: Enable RoPE positional embeddings (primarily for DNA sequences)
    """
    def __init__(self, n_embd, n_head, block_size, dropout, use_rope=False):
        super().__init__()
        head_size = n_embd // n_head
        self.n_head = n_head
        self.head_size = head_size
        self.key = nn.Linear(n_embd, n_embd)
        self.query = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(n_embd, n_embd)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(n_embd, n_embd)
        self.flash_available = hasattr(F, 'scaled_dot_product_attention')

        # Only initialize RoPE if needed (for DNA)
        self.use_rope = use_rope
        if use_rope:
            self.rotary = RotaryEmbedding(head_size)

        if self.flash_available:
            print(f"Using Flash Attention {'with RoPE' if use_rope else ''}")

    def apply_rotary_pos_emb(self, q, k, cos, sin):
        # Apply rotary embeddings to queries and keys
        q_embed = (q * cos) + (rotate_half(q) * sin)
        k_embed = (k * cos) + (rotate_half(k) * sin)
        return q_embed, k_embed

    def forward(self, x, mask=None):
        B, T, C = x.shape

        # Split heads and prepare q, k, v - ensure device consistency
        k = self.key(x).view(B, T, self.n_head, self.head_size).transpose(1, 2)
        q = self.query(x).view(B, T, self.n_head, self.head_size).transpose(1, 2)
        v = self.value(x).view(B, T, self.n_head, self.head_size).transpose(1, 2)

        # Apply RoPE only for DNA sequences
        if self.use_rope:
            cos, sin = self.rotary(x, seq_len=T)  # Pass x for device info
            q, k = self.apply_rotary_pos_emb(q, k, cos.to(x.device), sin.to(x.device))

        if self.flash_available:
            causal_mask = self.tril[:T, :T].bool()
            if mask is not None:
                if self.use_rope:
                    combined_mask = mask
                else:
                    combined_mask = torch.logical_and(
                        causal_mask.unsqueeze(0),
                        mask
                    )
            else:
                combined_mask = causal_mask.unsqueeze(0)

            attention_mask = combined_mask.float()
            attention_mask = attention_mask.masked_fill(~combined_mask, float('-inf'))
            attention_mask = attention_mask.unsqueeze(1)

            # Use flash attention with the correctly shaped mask
            if self.flash_available:
                # Check if our PyTorch version supports the advanced args
                try:
                    # Try with all optimizations
                    y = F.scaled_dot_product_attention(
                        q, k, v,
                        attn_mask=attention_mask,  # Shape: [B, 1, T, T]
                        dropout_p=self.dropout.p if self.training else 0.0,
                        is_causal=False,  # We're handling causality in our mask
                        scale=1.0 / math.sqrt(k.size(-1)),  # Explicit scaling for precision
                        mem_efficient=True  # Use memory efficient attention
                    )
                except TypeError:
                    # Fallback to standard arguments
                    y = F.scaled_dot_product_attention(
                        q, k, v,
                        attn_mask=attention_mask,  # Shape: [B, 1, T, T]
                        dropout_p=self.dropout.p if self.training else 0.0,
                        is_causal=False  # We're handling causality in our mask
                    )
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
            if mask is not None:
                att = att.masked_fill(mask[:, :T, :T].unsqueeze(1) == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.dropout(att)
            y = att @ v

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.proj(y)
        return y


def rotate_half(x):
    x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
    return torch.cat((-x2, x1), dim=-1)


class RotaryEmbedding(nn.Module):
    """
    Rotary Position Embeddings (RoPE) for handling very long sequences
    - More effective than absolute positional embeddings for long sequences
    - Allows model to extrapolate to longer sequences than seen during training
    """
    def __init__(self, dim, max_seq_len=32768, base=10000):
        super().__init__()
        inv_freq = 1. / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None
        self.max_seq_len = max_seq_len

    def forward(self, x, seq_len=None):
        if seq_len != self.seq_len_cached:
            self.seq_len_cached = seq_len
            t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
            # Ensure inv_freq is on the same device as x
            inv_freq = self.inv_freq.to(x.device)
            freqs = torch.einsum('i,j->ij', t, inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)

            # Scale positions for longer sequences
            if seq_len > self.max_seq_len:
                scale = math.log(seq_len / self.max_seq_len) + 1
                emb = emb / scale

            # Store cached values on the same device as input
            self.cos_cached = emb.cos()[None, None, :, :].to(x.device)
            self.sin_cached = emb.sin()[None, None, :, :].to(x.device)
        else:
            # Ensure cached values are on the correct device
            self.cos_cached = self.cos_cached.to(x.device)
            self.sin_cached = self.sin_cached.to(x.device)

        return self.cos_cached, self.sin_cached


class MultiQueryAttention(nn.Module):
    """
    Multi-Query Attention (GQA - Grouped Query Attention) with shared Key-Value heads.

    Implements true Grouped Query Attention where multiple query heads share the same
    key and value heads, providing optimal efficiency for backgammon move prediction.
    Uses 4:1 ratio (n_head=8, n_kv_heads=2) for backgammon game efficiency.

    GQA Benefits for Backgammon Training:
    - 2-3x faster attention computation per batch vs standard MHA
    - Better convergence per 1-3 epoch training session
    - Critical for short training sessions with limited time
    - Maintains attention quality for complex backgammon pattern recognition
    - Reduced memory footprint while preserving backgammon understanding

    Backgammon-Specific Advantages:
    - Efficient handling of game boundary masking
    - Optimal for autoregressive move prediction
    - Balances performance and memory for GPU training
    - Enables larger batches in short training sessions

    Args:
        n_embd: Embedding dimension (must be divisible by n_head)
        n_head: Number of query heads (attention outputs)
        n_kv_heads: Number of shared key/value heads (n_head // 4 = 4:1 GQA ratio)
        dropout: Dropout probability for attention weights
    """
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
        self.register_buffer('causal_mask', torch.tril(torch.ones(1024, 1024)))
        self.flash_available = hasattr(F, 'scaled_dot_product_attention')

        if self.flash_available:
            print("Using Flash Attention in MultiQueryAttention")

    def forward(self, x, mask=None):
        B, T, C = x.size()

        # Project queries
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # Project keys and values together
        kv = self.kv_proj(x).view(B, T, self.n_kv_heads, 2, self.head_dim)
        kv = kv.transpose(1, 2)
        k, v = kv[..., 0, :], kv[..., 1, :]

        # Repeat keys and values to match the number of query heads
        k = k.repeat_interleave(self.n_heads // self.n_kv_heads, dim=1)
        v = v.repeat_interleave(self.n_heads // self.n_kv_heads, dim=1)

        if self.flash_available:
            # Prepare masks
            causal_mask = self.causal_mask[:T, :T].bool()
            if mask is not None:
                game_mask = mask[:, :T, :T].bool()
                combined_mask = torch.logical_and(
                    causal_mask.unsqueeze(0),
                    game_mask
                )
            else:
                combined_mask = causal_mask.unsqueeze(0)

            attention_mask = combined_mask.unsqueeze(1)

            # Use flash attention
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
            # Fallback attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.causal_mask[:T, :T] == 0, float('-inf'))
            if mask is not None:
                att = att.masked_fill(mask[:, :T, :T].unsqueeze(1) == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.dropout(att)
            y = att @ v

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.out_proj(y)
        return y


class SwiGLU(nn.Module):
    """
    SwiGLU (Swish-Gated Linear Unit) activation function.

    A gated activation function that combines Swish (SiLU) gating with linear transformations.
    Provides better gradient flow and representation capacity compared to standard ReLU
    or GELU activations, commonly used in modern transformer architectures.

    Formula: SwiGLU(x) = (SiLU(W1*x) ⊙ W2*x) @ W3
    where ⊙ is element-wise multiplication

    Advantages over ReLU/GELU:
    - Better gradient flow through gating mechanism
    - Increased model capacity without parameter explosion
    - Improved performance on complex tasks like backgammon

    Args:
        in_features: Input feature dimension
        hidden_features: Hidden dimension for gating (default: 4*in_features)
    """
    def __init__(self, in_features, hidden_features=None):
        super().__init__()
        hidden_features = hidden_features or in_features * 4
        self.w1 = nn.Linear(in_features, hidden_features)  # Gate projection
        self.w2 = nn.Linear(in_features, hidden_features)  # Value projection
        self.w3 = nn.Linear(hidden_features, in_features)  # Output projection

    def forward(self, x):
        gate = F.silu(self.w1(x))  # SiLU activation for gating
        hidden = self.w2(x)        # Linear transformation for values
        return self.w3(gate * hidden)  # Gated combination and output projection


class BackgammonBlock(nn.Module):
    """
    Backgammon-optimized transformer block with MultiQueryAttention and SwiGLU.

    A complete transformer decoder block designed specifically for backgammon move prediction.
    Uses RMSNorm for efficient normalization, MultiQueryAttention for memory efficiency,
    and SwiGLU activation for better gradient flow. Includes residual connections
    and dropout for training stability.

    Architecture:
    - RMSNorm pre-attention normalization
    - MultiQueryAttention with game boundary masking
    - Residual connection + dropout
    - RMSNorm pre-feedforward normalization
    - SwiGLU feed-forward network
    - Residual connection + dropout

    This block is optimized for backgammon where attention patterns are complex but
    memory efficiency and gradient flow are critical.

    Args:
        n_embd: Embedding dimension (model width)
        n_head: Number of query attention heads
        n_kv_heads: Number of shared key/value heads
        dropout: Dropout probability for residual connections
    """
    def __init__(self, n_embd, n_head, n_kv_heads, dropout):
        super().__init__()
        self.rms_1 = RMSNorm(n_embd)
        self.attn = MultiQueryAttention(n_embd, n_head, n_kv_heads, dropout)
        self.rms_2 = RMSNorm(n_embd)
        self.swiglu = SwiGLU(n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Gradient checkpointing for memory efficiency in short training sessions
        if self.training:
            # Checkpoint attention layer to save memory during training
            def attn_checkpoint(attn_layer, rms_x, mask):
                return attn_layer(rms_x, mask=mask)

            def ffwd_checkpoint(ffwd_layer, rms_x):
                return ffwd_layer(rms_x)

            x = x + self.dropout(torch.utils.checkpoint.checkpoint(
                attn_checkpoint, self.attn, self.rms_1(x), mask, use_reentrant=False
            ))
            x = x + self.dropout(torch.utils.checkpoint.checkpoint(
                ffwd_checkpoint, self.swiglu, self.rms_2(x), use_reentrant=False
            ))
        else:
            # Normal forward pass during inference (no checkpointing needed)
            x = x + self.dropout(self.attn(self.rms_1(x), mask=mask))
            x = x + self.dropout(self.swiglu(self.rms_2(x)))
        return x


class TransformerBlock(nn.Module):
    def __init__(self, n_embd, n_head, block_size, dropout, use_dna=False):
        super().__init__()
        # Enable RoPE for DNA sequences
        self.sa = MultiHeadAttention(
            n_embd=n_embd,
            n_head=n_head,
            block_size=block_size,
            dropout=dropout,
            use_rope=use_dna  # Only use RoPE for DNA
        )
        self.ffwd = FeedForward(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x, mask=None):
        x = x + self.sa(self.ln1(x), mask=mask)
        x = x + self.ffwd(self.ln2(x))
        return x


class TransformerModel(nn.Module):
    def __init__(self, vocab_size, n_embd, n_head, block_size, n_layer, dropout, pretrained_embeddings=None, use_chess=False, use_dna=False):
        super().__init__()
        dtype = torch.get_default_dtype()
        if pretrained_embeddings is not None:
            self.token_embedding_table = nn.Embedding.from_pretrained(pretrained_embeddings, freeze=False)
            vocab_size, n_embd = pretrained_embeddings.shape
        else:
            self.token_embedding_table = nn.Embedding(vocab_size, n_embd, dtype=dtype)
        self.position_embedding_table = nn.Embedding(block_size, n_embd, dtype=dtype)
        self.blocks = nn.Sequential(*[TransformerBlock(n_embd, n_head, block_size, dropout, use_dna=use_dna) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
        self.block_size = block_size
        self.use_chess = use_chess
        self.use_dna = use_dna
        if use_chess:
            self.start_game_token = move_to_idx['<STARTGAME>']

    def create_mask(self, idx):
        if self.use_chess:
            # Existing chess game mask
            mask = torch.ones_like(idx, dtype=torch.float32)
            game_boundaries = (idx == self.start_game_token).float().cumsum(dim=1)
            mask = (game_boundaries.unsqueeze(1) == game_boundaries.unsqueeze(2)).float()
            return mask
        return None

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            # Masked loss: only count tokens that are move tokens (start with 'm_')
            if hasattr(self, 'is_move_vec') and self.is_move_vec is not None:
                per_tok = F.cross_entropy(logits, targets, reduction='none')
                with torch.no_grad():
                    mask = self.is_move_vec.to(targets.device)[targets]
                denom = torch.clamp(mask.sum(), min=1.0)
                loss = (per_tok * mask).sum() / denom
            else:
                loss = F.cross_entropy(logits, targets)

        return logits, loss


class BackgammonModel(nn.Module):
    """
    Backgammon move prediction transformer model with optimized architecture.

    A complete transformer model specifically designed for backgammon move prediction.
    Uses backgammon-specific optimizations including game boundary masking, MultiQueryAttention
    for efficiency, and RMSNorm for stable training. Based on proven architecture
    adapted for backgammon game dynamics.

    Architecture:
    - Backgammon move token embeddings + positional embeddings
    - Stack of BackgammonBlock layers (MultiQueryAttention + SwiGLU)
    - RMSNorm final normalization
    - Linear head for move prediction

    Key backgammon-specific features:
    - Game boundary masking prevents attention across game boundaries
    - Move tokenization captures backgammon-specific patterns (dice + moves)
    - MultiQueryAttention balances performance and memory efficiency
    - Optimized for backgammon strategy prediction

    Args:
        vocab_size: Size of backgammon move vocabulary
        n_embd: Embedding dimension (model width)
        n_head: Number of query attention heads
        n_kv_heads: Number of shared key/value heads
        block_size: Maximum sequence length (backgammon game length)
        n_layer: Number of transformer blocks
        dropout: Dropout probability for regularization
        use_chess: Enable backgammon-specific masking (always True for BackgammonModel)
        use_dna: Enable DNA-specific features (always False for BackgammonModel)
        use_value_head: Extra 2-way head predicting who won (White / Black) from the moves so far
    """
    def __init__(self, vocab_size, n_embd, n_head, n_kv_heads, block_size, n_layer, dropout, use_chess=False, use_dna=False, use_value_head=False):
        super().__init__()
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.use_chess = use_chess  # Using chess variable name for backgammon masking
        self.use_dna = use_dna
        self.use_value_head = use_value_head
        self.loser_move_weight = BACKGAMMON_DEFAULTS['loser_move_weight']
        self.value_loss_weight = BACKGAMMON_DEFAULTS['value_loss_weight']
        self.special_ids = {}
        if use_chess:
            self.start_game_token = None  # Will be set after vocab creation

        # Standard embeddings
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)

        # Use BackgammonBlock with true GQA (Grouped Query Attention) - 4:1 ratio for efficiency
        # n_head=8, n_kv_heads=2 provides 2-3x faster attention than standard MHA
        self.blocks = nn.ModuleList([
            BackgammonBlock(
                n_embd=n_embd,
                n_head=n_head,
                n_kv_heads=n_kv_heads,  # 4:1 GQA ratio (n_head // 4) for optimal backgammon training
                dropout=dropout
            ) for _ in range(n_layer)
        ])

        # Final RMSNorm instead of LayerNorm
        self.rms_final = RMSNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

        # Value head (Sep 2026): 0 = White won (<W>), 1 = Black won (<B>).
        # Trained only on games whose input result token was masked to <U>, so it cannot cheat.
        if self.use_value_head:
            self.head_value = nn.Linear(n_embd, 2)

        # Initialize weights
        self.apply(self._init_weights)

    def set_special_ids(self, move_to_idx):
        """Remember ids the loss and the value head need. Safe to call again after the vocab grows."""
        self.special_ids = {}
        for name in ('<STARTGAME>', '<EOFG>', '<EOM>', '<PAD>', '<W>', '<B>', '<U>'):
            if name in move_to_idx:
                self.special_ids[name] = move_to_idx[name]
        if '<STARTGAME>' in move_to_idx:
            self.start_game_token = move_to_idx['<STARTGAME>']

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def create_game_mask(self, idx):
        """Create attention mask for backgammon games"""
        if not self.use_chess:
            return None
        mask = torch.ones_like(idx, dtype=torch.float32)
        game_boundaries = (idx == self.start_game_token).float().cumsum(dim=1)
        mask = (game_boundaries.unsqueeze(1) == game_boundaries.unsqueeze(2)).float()
        return mask

    def _backbone(self, idx):
        """Embeddings + transformer blocks + final norm. Returns hidden states [B, T, n_embd]."""
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))
        x = tok_emb + pos_emb
        for block in self.blocks:
            x = block(x, mask=self.create_game_mask(idx))
        return self.rms_final(x)

    def game_start_index(self, idx):
        """For every position t, index of the most recent <STARTGAME> at or before t. Shape [B, T]."""
        B, T = idx.shape
        t = torch.arange(T, device=idx.device).unsqueeze(0).expand(B, T)
        is_start = (idx == self.start_game_token)
        start_idx = torch.where(is_start, t, torch.zeros_like(t)).cummax(dim=1).values
        return start_idx

    def _value_loss(self, h, idx, targets):
        """Cross-entropy of the win head. Only games whose input result token is <U>.
        The true winner is the target at <STARTGAME> (y still has <W> or <B> after <U> masking).
        """
        if not self.use_value_head:
            return None
        sp = self.special_ids
        if '<U>' not in sp or '<W>' not in sp or '<B>' not in sp or self.start_game_token is None:
            return None
        B, T, _ = h.shape
        start_idx = self.game_start_index(idx)
        true_result = targets.gather(1, start_idx)
        input_result = idx.gather(1, (start_idx + 1).clamp(max=T - 1))
        q = torch.arange(T, device=idx.device).unsqueeze(0) + 1 - start_idx
        value_tgt = torch.full_like(true_result, -1)
        value_tgt = torch.where(true_result == sp['<W>'], torch.zeros_like(value_tgt), value_tgt)
        value_tgt = torch.where(true_result == sp['<B>'], torch.ones_like(value_tgt), value_tgt)
        pad_id = sp.get('<PAD>')
        not_pad = torch.ones_like(targets, dtype=torch.bool) if pad_id is None else (targets != pad_id)
        value_mask = (input_result == sp['<U>']) & (value_tgt >= 0) & (q >= 2) & not_pad
        if not value_mask.any():
            return None
        logits = self.head_value(h[value_mask])
        return F.cross_entropy(logits, value_tgt[value_mask])

    @torch.no_grad()
    def predict_value(self, idx):
        """Softmax over (White won, Black won) at the last position. Prompt should start <STARTGAME> <U>."""
        if not self.use_value_head:
            return None
        h = self._backbone(idx)
        return F.softmax(self.head_value(h[:, -1]), dim=-1)

    def forward(self, idx, targets=None, token_weights=None):
        B, T = idx.shape

        # Hidden states are shared by the move head and the win head
        x = self._backbone(idx)
        logits = self.lm_head(x)

        # Calculate loss if training
        if targets is None:
            loss = None
            return logits, loss
        else:
            logits_flat = logits.view(B*T, -1)
            targets_flat = targets.view(B*T)
            # token_weights: 0.5 on the loser's checker moves, 1 everywhere else (from the dataset)
            if token_weights is None:
                token_weights = torch.ones(B, T, device=idx.device, dtype=torch.float32)
            else:
                token_weights = token_weights.to(device=idx.device, dtype=torch.float32)
            # Masked loss: only count move tokens if mask is available
            if hasattr(self, 'is_move_vec') and self.is_move_vec is not None:
                per_tok = F.cross_entropy(logits_flat, targets_flat, reduction='none')
                with torch.no_grad():
                    mask = self.is_move_vec.to(targets_flat.device)[targets_flat]
                w = mask * token_weights.reshape(-1)
                # Keep the count on device. .item() here splits torch.compile and does not change the loss.
                # Zero move tokens already give loss 0: numerator is 0, clamp keeps the denominator at 1.
                num_move_tokens = (mask > 0).sum()
                loss = (per_tok * w).sum() / w.sum().clamp(min=1.0)
            else:
                per_tok = F.cross_entropy(logits_flat, targets_flat, reduction='none')
                w = token_weights.reshape(-1)
                loss = (per_tok * w).sum() / w.sum().clamp(min=1.0)
                num_move_tokens = B * T

            loss_value = self._value_loss(x, idx, targets)
            if loss_value is not None:
                loss = loss + self.value_loss_weight * loss_value

            # Return loss and number of move tokens for correct multi-GPU averaging
            return logits, (loss, num_move_tokens)


# Backgammon dataset and utility functions
class BackgammonMovesDataset(Dataset):
    """
    Dataset for backgammon games using split tokenization.

    Processes backgammon games using split tokenization where moves are separated
    into dice and move components (e.g., "d41 m_lpab", "d61 m_mghg").
    Supports parallel processing for large datasets and includes validation.

    Tokenization process:
    - Split text into individual games (blank-line separated)
    - Add <STARTGAME> and <EOFG> boundary markers to each game
    - Tokenize using split format: dice tokens (d11-d66) + move tokens (m_xxxx)
    - Convert tokens to integer indices using vocabulary mapping
    - Validate all tokens are within vocabulary range
    - Create overlapping sequences for next-token prediction

    Benefits of split tokenization:
    - Better learning of dice-move relationships
    - Smaller, more efficient models possible
    - Cleaner semantic understanding

    Parallel processing:
    - For large datasets (>1M chars), uses multiprocessing for speed
    - Splits data into chunks and processes concurrently
    - Recombines results and validates integrity

    Args:
        text: Raw backgammon game text with split tokens, games separated by blank lines
        seq_length: Length of each training sequence (context window)
        move_to_idx: Dictionary mapping backgammon tokens to integer indices
    """
    def __init__(self, text, seq_length, move_to_idx):
        self.seq_length = seq_length
        self.move_to_idx = move_to_idx  # Store for game boundary detection
        self.tokens = []

        # Precompile common string patterns for faster matching
        self.start_game_pattern = '<STARTGAME>'
        self.eofg_pattern = '<EOFG>'

        # For large datasets, use parallel processing to speed up tokenization.
        # This step is CPU-only. nvidia-smi stays near 0% until the training loop starts.
        if len(text) > 1_000_000:  # Only parallelize for large datasets
            print("Tokenizing games on the CPU (GPU idle until the first training batch)...")
            import multiprocessing as mp
            from concurrent.futures import ProcessPoolExecutor

            # Split text into chunks for parallel processing
            chunk_size = 200_000  # Adjust based on your system
            chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

            # Process chunks in parallel
            num_cores = mp.cpu_count() - 1  # Leave one core free for system
            with ProcessPoolExecutor(max_workers=num_cores) as executor:
                # Pass all required arguments as a tuple
                chunk_args = [(chunk, self.start_game_pattern, self.eofg_pattern, move_to_idx)
                              for chunk in chunks]
                results = list(executor.map(process_chunk_for_backgammon_moves, chunk_args))

            # Combine all results
            for chunk_tokens in results:
                self.tokens.extend(chunk_tokens)

            print(f"Parallel tokenization complete with {num_cores} cores")
        else:
            # Original sequential tokenization for smaller datasets
            i = 0
            while i < len(text):
                if text[i:i+11] == self.start_game_pattern:
                    self.tokens.append(move_to_idx[self.start_game_pattern])
                    i += 11
                elif text[i:i+6] == self.eofg_pattern:
                    self.tokens.append(move_to_idx[self.eofg_pattern])
                    i += 6
                elif text[i].isspace():
                    # Skip spaces
                    i += 1
                else:
                    # Extract split tokens (like "d41", "m_lpab")
                    token_start = i
                    while i < len(text) and not text[i].isspace():
                        i += 1
                    if token_start < i:
                        token = text[token_start:i]
                        if token in move_to_idx:
                            self.tokens.append(move_to_idx[token])
                        else:
                            # Skip invalid token sequences
                            pass

        # Validate all tokens are within vocabulary range
        vocab_size = len(move_to_idx)
        invalid_tokens = [token for token in self.tokens if token >= vocab_size or token < 0]
        if invalid_tokens:
            print(f"Warning: Found {len(invalid_tokens)} invalid tokens, replacing with <PAD>")
            pad_token = move_to_idx['<PAD>']
            self.tokens = [token if 0 <= token < vocab_size else pad_token for token in self.tokens]

        # Convert tokens to a tensor for faster indexing during training
        self.tokens_tensor = torch.tensor(self.tokens, dtype=torch.long)

        # Final validation - ensure tensor values are in valid range
        valid_mask = (self.tokens_tensor >= 0) & (self.tokens_tensor < vocab_size)
        if not valid_mask.all():
            invalid_count = (~valid_mask).sum().item()
            print(f"Final validation: Found {invalid_count} invalid tokens in tensor, clamping to valid range")
            self.tokens_tensor = torch.clamp(self.tokens_tensor, 0, vocab_size - 1)

        print(f"Tokenized {len(self.tokens)} backgammon moves, all validated")

        # Who moved first in each game, in the same order as <STARTGAME> tokens.
        # <1W>/<1B> is written by Backgammon_SGF_to_TXT and is not a model token.
        self._first_movers = self._scan_first_movers(text)
        self._w_id = move_to_idx.get('<W>')
        self._b_id = move_to_idx.get('<B>')
        self._u_id = move_to_idx.get('<U>')
        self._eom_id = move_to_idx.get('<EOM>')
        self._move_token_ids = {i for t, i in move_to_idx.items() if isinstance(t, str) and t.startswith('m_')}
        self._loser_move_weight = BACKGAMMON_DEFAULTS['loser_move_weight']
        self._value_mask_prob = BACKGAMMON_DEFAULTS['value_mask_prob']

    def _scan_first_movers(self, text):
        """1 = White moved first, 2 = Black moved first, 0 = unlabeled game."""
        movers = []
        for match in re.finditer(r'<STARTGAME>', text):
            window = text[match.end():match.end() + 48]
            found = re.match(r'\s*(?:<W>|<B>)?\s*(<1W>|<1B>)?', window)
            if found and found.group(1) == '<1W>':
                movers.append(1)
            elif found and found.group(1) == '<1B>':
                movers.append(2)
            else:
                movers.append(0)
        return movers

    def __len__(self):
        # Find all STARTGAME positions for sequence starts (cache this)
        if not hasattr(self, '_game_starts'):
            startgame_token = self.move_to_idx['<STARTGAME>']
            # Use torch operations for speed (avoid python loop over millions of tokens)
            self._game_starts = torch.nonzero(self.tokens_tensor == startgame_token, as_tuple=False).flatten().tolist()
        return len(self._game_starts)

    def __getitem__(self, idx):
        # Start sequence at game boundary
        start_pos = self._game_starts[idx]

        # Calculate how much data is available for x and y
        len_file = len(self.tokens_tensor)

        # X source: start_pos ... start_pos + seq_len
        x_end = min(start_pos + self.seq_length, len_file)
        x_data = self.tokens_tensor[start_pos : x_end]

        # Y source: start_pos + 1 ... start_pos + seq_len + 1
        y_end = min(start_pos + self.seq_length + 1, len_file)
        y_data = self.tokens_tensor[start_pos + 1 : y_end]

        # Initialize full padded tensors
        pad_token = self.move_to_idx['<PAD>']
        x = torch.full((self.seq_length,), pad_token, dtype=torch.long)
        y = torch.full((self.seq_length,), pad_token, dtype=torch.long)

        # Copy available data into padded tensors
        x[:len(x_data)] = x_data
        y[:len(y_data)] = y_data

        # Per-target weight. 0.5 on checker moves by the side that lost, 1 otherwise.
        # Built from the true tokens (before <U> masking) so the winner is still known.
        # A short game can run into the next game inside one window; each <STARTGAME> starts over.
        token_weights = torch.ones(self.seq_length, dtype=torch.float32)
        start_id = self.move_to_idx['<STARTGAME>']
        i = 0
        game_i = idx
        while i < len(x_data):
            if int(x_data[i]) != start_id:
                i += 1
                continue
            first = self._first_movers[game_i] if game_i < len(self._first_movers) else 0
            has_result = (i + 1 < len(x_data) and self._w_id is not None
                          and int(x_data[i + 1]) in (self._w_id, self._b_id) and first in (1, 2))
            if has_result:
                winner_is_white = int(x_data[i + 1]) == self._w_id
                side_is_white = (first == 1)  # turn 0 is whoever moved first; <EOM> flips the side
                j = i + 2
                while j < len(x_data) and int(x_data[j]) != start_id:
                    tok_id = int(x_data[j])
                    if tok_id == self._eom_id:
                        side_is_white = not side_is_white
                    elif tok_id in self._move_token_ids and side_is_white != winner_is_white:
                        # y[j-1] is the target that predicts token x_data[j]
                        token_weights[j - 1] = self._loser_move_weight
                    j += 1
                i = j
            else:
                j = i + 1
                while j < len(x_data) and int(x_data[j]) != start_id:
                    j += 1
                i = j
            game_i += 1

        # Hide the result token in the INPUT of some games. y keeps the true <W> or <B>.
        # Every game that starts inside this window is masked on its own coin flip.
        if self._u_id is not None and self._value_mask_prob > 0 and self._w_id is not None:
            for rel in range(len(x_data)):
                if int(x[rel]) != start_id:
                    continue
                if rel + 1 < len(x_data) and int(x[rel + 1]) in (self._w_id, self._b_id):
                    if random.random() < self._value_mask_prob:
                        x[rel + 1] = self._u_id

        return x, y, token_weights


def process_chunk_for_backgammon_moves(args):
    chunk_text, start_game_pattern, eofg_pattern, move_to_idx = args
    chunk_tokens = []
    i = 0
    while i < len(chunk_text):
        if chunk_text[i:i+11] == start_game_pattern:
            chunk_tokens.append(move_to_idx[start_game_pattern])
            i += 11
        elif chunk_text[i:i+6] == eofg_pattern:
            chunk_tokens.append(move_to_idx[eofg_pattern])
            i += 6
        elif chunk_text[i].isspace():
            i += 1
        else:
            # Extract split tokens (like "d41", "m_lpab")
            token_start = i
            while i < len(chunk_text) and not chunk_text[i].isspace():
                i += 1
            if token_start < i:
                token = chunk_text[token_start:i]
                if token in move_to_idx:
                    chunk_tokens.append(move_to_idx[token])
    return chunk_tokens


def create_move_to_idx_from_text(text):
    """
    Create backgammon vocabulary from training text with split tokenization.

    Uses split tokenization where moves are separated into dice and move components:
    - "d11", "d12", ..., "d66" for dice rolls
    - "m_adln", "m_mhxw", etc. for move sequences

    This enables better learning of dice-move relationships and more efficient training.

    Args:
        text: Raw backgammon training text with split tokens

    Returns:
        Dictionary mapping token strings to integer indices
    """
    # Start with special tokens
    token_to_idx = {}
    special_tokens = ['<STARTGAME>', '<EOFG>', '<EOM>', '<NOMOVE>', '<PAD>']

    for idx, token in enumerate(special_tokens):
        token_to_idx[token] = idx

    # Unique tokens only. Do not collect every token in the file: an 8 GB games file
    # would build a list of billions of strings and never reach the GPU.
    print("Scanning unique tokens (CPU only — the GPU stays idle until training starts)...")
    import re
    unique_tokens = set()
    for match in re.finditer(r'[^ \n<>]+', text):
        token = match.group()
        if token and not token.startswith('<'):
            unique_tokens.add(token)

    # Categorize tokens for analysis
    dice_tokens = set()
    move_tokens = set()
    other_tokens = set()

    for token in unique_tokens:
        if re.match(r'^d\d+$', token):  # d11, d22, etc.
            dice_tokens.add(token)
        elif token.startswith('m_'):  # m_adln, m_mhxw, etc.
            move_tokens.add(token)
        else:
            other_tokens.add(token)

    # Sort for consistent ordering
    sorted_tokens = sorted(unique_tokens)

    # Add tokens to vocabulary
    for token in sorted_tokens:
        if token not in token_to_idx:  # Don't overwrite special tokens
            token_to_idx[token] = len(token_to_idx)

    print("🎯 SPLIT TOKENIZATION: Dice + Move tokens!")
    print(f"   - Special tokens: {len(special_tokens)}")
    print(f"   - Dice tokens: {len(dice_tokens)} (d11-d66)")
    print(f"   - Move tokens: {len(move_tokens)} (m_xxxx format)")
    print(f"   - Other tokens: {len(other_tokens)}")
    print(f"   - TOTAL VOCAB: {len(token_to_idx)} tokens")
    print(f"   - Learning benefit: Model understands dice-move relationships!")

    # <W>, <B>, <U> go at the end so an older checkpoint's token indices stay put.
    ensure_result_tokens(token_to_idx)
    return token_to_idx


def ensure_result_tokens(move_to_idx):
    """Append <W>, <B>, <U> if they are missing. Existing indices are not changed.

    <W> white won, <B> black won, <U> result hidden from the model on some inputs.
    Returns how many tokens were added.
    """
    added = 0
    for tok in ('<W>', '<B>', '<U>'):
        if tok not in move_to_idx:
            move_to_idx[tok] = len(move_to_idx)
            added += 1
    if added:
        print(f"Added result tokens <W> <B> <U> ({added} new). Older token indices unchanged.")
    return added


def create_idx_to_move(move_to_idx):
    idx_to_move = {idx: move for move, idx in move_to_idx.items()}
    return idx_to_move


def merge_vocabularies(existing_vocab, new_text):
    """
    Merge new tokens from text into existing vocabulary without changing existing token indices.
    
    This allows continuing training with new datasets that may contain new tokens.
    Existing tokens keep their original indices, new tokens are added at the end.
    
    Args:
        existing_vocab: Dictionary mapping token strings to integer indices (from checkpoint)
        new_text: Raw backgammon training text that may contain new tokens
        
    Returns:
        Dictionary mapping token strings to integer indices (merged vocabulary)
        Number of new tokens added
    """
    import re
    
    # Start with existing vocabulary (preserves all existing token indices)
    merged_vocab = existing_vocab.copy()
    original_size = len(merged_vocab)
    
    # Extract all unique tokens from the new text
    tokens = re.findall(r'(?:^| )([^ <][^ ]*?)(?= |$)', new_text)
    
    # Find tokens that are not in existing vocabulary
    unique_new_tokens = set()
    for token in tokens:
        token = token.strip()
        if token and not token.startswith('<') and token not in merged_vocab:
            unique_new_tokens.add(token)
    
    # Sort new tokens for consistent ordering
    sorted_new_tokens = sorted(unique_new_tokens)
    
    # Add new tokens to vocabulary (they get indices after existing tokens)
    for token in sorted_new_tokens:
        merged_vocab[token] = len(merged_vocab)

    # Result tokens are not in the game text as raw vocab items the regex keeps,
    # so append them here too. Older checkpoints gain them without shifting indices.
    added_results = ensure_result_tokens(merged_vocab)
    
    new_token_count = len(merged_vocab) - original_size

    if new_token_count > 0:
        print(f"🔄 Vocabulary merge: Added {new_token_count} new tokens")
        print(f"   - Original vocab size: {original_size}")
        print(f"   - New vocab size: {len(merged_vocab)}")
        print(f"   - Result tokens added this merge: {added_results}")
        print(f"   - All existing token indices preserved ✅")
    else:
        print(f"✅ Vocabulary merge: No new tokens found (all tokens already in vocabulary)")
    
    return merged_vocab, new_token_count


def expand_model_embeddings(model, old_vocab_size, new_vocab_size, n_embd):
    """
    Expand model's embedding layer to accommodate new vocabulary tokens.
    
    Preserves existing token embeddings and initializes new token embeddings randomly.
    Also expands the language model head (lm_head) to match new vocab size.
    
    Args:
        model: BackgammonModel instance
        old_vocab_size: Original vocabulary size (from checkpoint)
        new_vocab_size: New vocabulary size (after merging)
        n_embd: Embedding dimension
    """
    if new_vocab_size <= old_vocab_size:
        print(f"✅ No embedding expansion needed (vocab size: {old_vocab_size} → {new_vocab_size})")
        return
    
    print(f"🔧 Expanding model embeddings: {old_vocab_size} → {new_vocab_size} tokens")
    
    # Get existing embeddings
    old_embeddings = model.token_embedding_table.weight.data.clone()
    
    # Create new embedding layer with larger vocabulary
    new_embeddings = nn.Embedding(new_vocab_size, n_embd, dtype=old_embeddings.dtype)
    
    # Copy existing embeddings (preserves learned representations)
    new_embeddings.weight.data[:old_vocab_size] = old_embeddings
    
    # Initialize new token embeddings randomly (standard normal initialization)
    # Using same initialization as PyTorch default for Embedding layers
    nn.init.normal_(new_embeddings.weight.data[old_vocab_size:], mean=0.0, std=0.02)
    
    # Replace embedding layer (preserve device)
    device = old_embeddings.device
    model.token_embedding_table = new_embeddings.to(device)
    
    # Expand language model head (lm_head) to match new vocab size
    old_lm_head = model.lm_head.weight.data.clone()
    old_lm_bias = model.lm_head.bias.data.clone() if model.lm_head.bias is not None else None
    new_lm_head = nn.Linear(n_embd, new_vocab_size, dtype=old_lm_head.dtype)
    
    # Copy existing weights
    new_lm_head.weight.data[:old_vocab_size] = old_lm_head
    
    # Initialize new weights randomly
    nn.init.normal_(new_lm_head.weight.data[old_vocab_size:], mean=0.0, std=0.02)
    if new_lm_head.bias is not None:
        if old_lm_bias is not None:
            new_lm_head.bias.data[:old_vocab_size] = old_lm_bias
        nn.init.zeros_(new_lm_head.bias.data[old_vocab_size:])
    
    # Replace lm_head (preserve device)
    model.lm_head = new_lm_head.to(device)
    
    # Update move token mask if it exists (expand with zeros for new tokens)
    if hasattr(model, 'is_move_vec'):
        old_mask = model.is_move_vec.clone()
        new_mask = torch.zeros(new_vocab_size, dtype=old_mask.dtype)
        new_mask[:old_vocab_size] = old_mask
        # New tokens will be checked and set appropriately below
        model.register_buffer('is_move_vec', new_mask)
    
    print(f"✅ Model embeddings expanded successfully")
    print(f"   - Existing {old_vocab_size} token embeddings preserved")
    print(f"   - New {new_vocab_size - old_vocab_size} token embeddings initialized randomly")


def filter_optimizer_state(optimizer_state_dict, model):
    """
    Filter optimizer state to remove entries for parameters that changed size.
    
    When vocabulary expands, embedding and lm_head parameters change size.
    Their optimizer state (momentum buffers) must be removed to prevent shape mismatches.
    
    CRITICAL: Only filter the 'state' dict, NOT param_groups. The param_groups structure
    must match the new optimizer, and the optimizer will skip missing state entries.
    
    Args:
        optimizer_state_dict: Optimizer state dict from checkpoint (can be dict or list)
        model: Current model with expanded parameters
        
    Returns:
        Filtered optimizer state dict/list with mismatched parameters removed from state only
    """
    if optimizer_state_dict is None:
        return None
    
    # Handle list of optimizer states (multi-GPU)
    if isinstance(optimizer_state_dict, list):
        return [filter_optimizer_state(opt_state, model) for opt_state in optimizer_state_dict]
    
    # Get current model parameter shapes
    model_param_shapes = {name: param.shape for name, param in model.named_parameters()}
    
    # Create filtered state dict - KEEP param_groups unchanged
    filtered_state = {
        'state': {},
        'param_groups': optimizer_state_dict.get('param_groups', [])  # Keep original param_groups
    }
    
    # Only filter the 'state' dict to remove mismatched parameter states
    for param_id, old_state in optimizer_state_dict.get('state', {}).items():
        # Get shape from optimizer state (momentum buffer)
        old_shape = None
        if 'exp_avg' in old_state:
            old_shape = old_state['exp_avg'].shape
        elif 'exp_avg_sq' in old_state:
            old_shape = old_state['exp_avg_sq'].shape
        
        # Check if any current model parameter matches this shape
        shape_matches = False
        if old_shape:
            for name, current_shape in model_param_shapes.items():
                if current_shape == old_shape:
                    shape_matches = True
                    break
        else:
            # No shape to check (e.g., only 'step' counter), keep it
            shape_matches = True
        
        # Only include state if shape matches
        if shape_matches:
            filtered_state['state'][param_id] = old_state
    
    return filtered_state


def optimizer_state_matches(optimizer, optimizer_state_dict):
    """True when the checkpoint optimizer still has one slot per model parameter.

    Adding the value head changes the parameter count. PyTorch will not load an
    optimizer state whose group size does not match, so the caller keeps a fresh
    optimizer instead of crashing.
    """
    state = optimizer_state_dict[0] if isinstance(optimizer_state_dict, list) else optimizer_state_dict
    if not isinstance(state, dict) or 'param_groups' not in state:
        return True
    saved = sum(len(g.get('params', [])) for g in state['param_groups'])
    current = sum(len(g['params']) for g in optimizer.param_groups)
    if saved != current:
        print(f"Optimizer state skipped: checkpoint has {saved} parameters, model has {current}.")
        print("   A new value head changes the count, so this resume uses a fresh optimizer.")
        return False
    return True


def load_backgammon_file():
    """Load backgammon games file for training"""
    print("Please select a backgammon games file.")
    file_path = filedialog.askopenfilename(
        title="Select Backgammon Games File",
        filetypes=[("Text files", "*.txt")]
    )

    if file_path:
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()
            print(f"Backgammon games file loaded: {file_path}")
            print(f"Total characters: {len(text)}")

            # Process games - concatenate for smarter GPU usage (like ChessBrain)
            games = text.split('\n\n')
            games = [game.strip() for game in games if game.strip()]
            text = '\n'.join(games)  # Single newlines for continuous text stream
            print(f"Concatenated {len(games)} backgammon games for efficient GPU training")

            return text, file_path
        except Exception as e:
            print(f"An error occurred: {e}")
            return None, None
    else:
        print("No file selected.")
        return None, None


def get_input_with_default(prompt, default_value):
    try:
        value = input(f"{prompt} (default: {default_value}): ")
        return value if value.strip() else default_value
    except EOFError:
        # Handle non-interactive environments
        print(f"{prompt} (default: {default_value}): {default_value}")
        return default_value


def create_file_dialog(title="Select File", filetypes=None, initialdir=None):
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title=title,
        filetypes=filetypes,
        initialdir=initialdir
    )
    root.destroy()
    return file_path


def save_token_embeddings(model, filepath):
    model_module = model.module if isinstance(model, nn.DataParallel) else model
    embeddings = model_module.token_embedding_table.weight.data
    torch.save(embeddings, filepath)
    filename = os.path.basename(filepath)
    model_folder = os.path.dirname(filepath)
    print(f"Token embedding saved: {filename} in {model_folder}")


def save_model_all(model, all_text, n_embd, n_head, n_kv_heads, n_layer, dropout, block_size, epoch, batch_idx, batch_size, optimizer, scheduler, scaler, loss, learning_rate=None, weight_decay=None, gpu_indices=None):
    """
    Save complete backgammon model checkpoint with all training state.

    Creates a comprehensive checkpoint containing model weights, training state,
    and generated samples. Optimized for backgammon model resumption and analysis.

    Saves:
    - Model architecture and trained weights
    - Optimizer state(s) (Adafactor) for training continuation
    - Learning rate scheduler state(s)
    - Gradient scaler state(s) for mixed precision
    - Training progress (epoch, batch, hyperparameters)
    - GPU configuration used for training
    - Sample generated backgammon moves for progress tracking
    - Token embeddings separately for analysis

    File naming: B{n_layer}H{n_head}E{n_embd}_B{batch_size}_E{epoch}B{batch}_L{loss}_{timestamp}.pth

    Args:
        optimizer: Single optimizer dict OR list of optimizer dicts (for multi-GPU)
        scheduler: Single scheduler dict OR list of scheduler dicts (for multi-GPU)
        scaler: Single scaler dict OR list of scaler dicts (for multi-GPU)
        (other args same as before)
    """
    timestamp = datetime.now().strftime("%m%d_%H%M")
    BASE_DIR = "/Users/jonathanrothberg/Data" if platform.system() == "Darwin" else "/home/jonathan/Data"

    # Ensure the base directory exists
    os.makedirs(BASE_DIR, exist_ok=True)

    model_prefix = "B"  # Backgammon LLM
    model_id = f"{model_prefix}{n_layer}H{n_head}E{n_embd}"
    if n_kv_heads != n_head:
        model_id += f"K{n_kv_heads}"

    folder_prefix = "Backgammon"
    model_folder = os.path.join(BASE_DIR, f"{folder_prefix}_Model_{model_id}")
    model_filename = f"{model_id}_B{batch_size}_E{epoch+1}B{batch_idx+1}_L{loss:.3f}_{timestamp}.pth"

    if not os.path.exists(model_folder):
        os.makedirs(model_folder)

    if isinstance(model, nn.DataParallel):
        model_state_dict = model.module.state_dict()
    else:
        model_state_dict = model.state_dict()

    # Handle both single objects and lists (for multi-GPU)
    if isinstance(optimizer, list):
        # Check if list contains optimizer objects or state dicts
        if len(optimizer) > 0 and hasattr(optimizer[0], 'state_dict'):
            # List of optimizer objects - call state_dict()
            optimizer_state = [opt.state_dict() for opt in optimizer]
        else:
            # List of state dicts (from checkpoint loading) - use directly
            optimizer_state = optimizer

        if len(scheduler) > 0 and hasattr(scheduler[0], 'state_dict'):
            scheduler_state = [sched.state_dict() for sched in scheduler]
        else:
            scheduler_state = scheduler

        if scaler and len(scaler) > 0:
            if hasattr(scaler[0], 'state_dict'):
                scaler_state = [scal.state_dict() for scal in scaler]
            else:
                scaler_state = scaler
        else:
            scaler_state = None
    else:
        optimizer_state = optimizer.state_dict()
        scheduler_state = scheduler.state_dict()
        scaler_state = scaler.state_dict() if scaler else None

    checkpoint = {
        'model_state_dict': model_state_dict,
        'optimizer_state_dict': optimizer_state,
        'scheduler_state_dict': scheduler_state,
        'scaler_state_dict': scaler_state,
        'epoch': epoch,
        'batch_idx': batch_idx,
        'hyperparameters': {
            'vocab_size': len(move_to_idx),
            'n_embd': n_embd,
            'n_head': n_head,
            'n_kv_heads': n_kv_heads,
            'n_layer': n_layer,
            'dropout': dropout,
            'block_size': block_size,
            'use_chess': True,  # Using chess variable for backgammon masking
            'use_dna': False,
            'has_value_head': True,  # White/Black win head is part of this trainer
            # Training parameters (can be changed when reloading)
            'batch_size': batch_size,
            'learning_rate': learning_rate if learning_rate is not None else 3e-4,  # Current learning rate
            'weight_decay': weight_decay if weight_decay is not None else 0.01,   # Current weight decay
            'gpu_indices': gpu_indices,   # GPUs used for training
        },
        'tokenizer': move_to_idx,
        'dataset_type': 'backgammon_moves'
    }

    print(f"\nModel folder: {model_folder}, {checkpoint['hyperparameters']}")
    torch.save(checkpoint, os.path.join(model_folder, model_filename))
    print(f"Model saved: {model_filename} in {model_folder}")

    filenamealltext = f"all_text_{epoch+1}_{timestamp}.txt"
    with open(os.path.join(model_folder, filenamealltext), 'w', encoding='utf-8') as file:
        file.write(all_text)
    print(f"Text saved: {filenamealltext} in {model_folder}")

    embedding_filename = f"token_embeddings_{timestamp}.pt"
    save_token_embeddings(model, os.path.join(model_folder, embedding_filename))


def test_progress(epoch, num_epochs, batch_idx, data_loader, loss, model, x, tokens_to_generate, all_text, idx_to_move):
    """
    Generate sample backgammon moves during training to monitor model progress.

    CORRECT IMPLEMENTATION: Uses top-K parallel ranking (same as game inference)
    - Single forward pass through model
    - Ranks ALL vocabulary tokens by probability
    - Returns top-K alternative moves for the SAME dice roll
    - Matches exactly what the game does during play
    
    Process:
    1. Switch model to evaluation mode (disables dropout)
    2. Unwrap DataParallel if present to access model methods
    3. Extract sample sequence from current batch
    4. Get top-K predictions via parallel ranking (NOT autoregressive)
    5. Format and display both input and predicted moves
    6. Accumulate samples for checkpoint saving

    Args:
        epoch: Current training epoch
        num_epochs: Total training epochs
        batch_idx: Current batch index within epoch
        data_loader: Training data loader (for progress display)
        loss: Current training loss value
        model: BackgammonModel (may be wrapped in DataParallel)
        x: Current batch input tensor [batch_size, seq_len]
        tokens_to_generate: Number of moves to generate (typically small, ~50)
        all_text: Accumulator string for generated samples across training
        idx_to_move: Dictionary mapping token indices back to backgammon moves

    Returns:
        Updated all_text string with new generated samples appended
    """
    print(f"\nEpoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(data_loader)}], Loss: {loss:.4f}")

    was_training = model.training
    model.eval()
    model_single = model.module if isinstance(model, nn.DataParallel) else model
    
    # Determine the correct device from the model parameters
    # This is robust against current_device changes (e.g. from set_device calls)
    try:
        model_device = next(model_single.parameters()).device
    except StopIteration:
        # Fallback if model has no parameters (unlikely)
        model_device = device

    with torch.no_grad():
        input_seq = x[-1].unsqueeze(0).to(model_device) # Ensure input is on model device
        # Safety check for input sequence display
        input_seq_str = ' '.join([idx_to_move.get(idx.item(), f'<UNK:{idx.item()}>') for idx in input_seq[0]])

        print("\nInput Sequence:")
        print(input_seq_str)

        # 🔧 FIX: Back up to the last dice roll and predict complete move sequences
        # Find the last dice tokens in the sequence
        tokens_list = [idx_to_move.get(idx.item(), '') for idx in input_seq[0]]
        game_history = []
        last_dice_tokens = []

        # Scan backwards to find the last complete dice roll
        i = len(tokens_list) - 1
        while i >= 0:
            token = tokens_list[i]
            if token.startswith('d') and len(token) == 2:  # Found a dice token like 'd3', 'd1'
                # Check if this is part of a dice pair
                if i > 0 and tokens_list[i-1].startswith('d') and len(tokens_list[i-1]) == 2:
                    # Found dice pair (doubles or regular)
                    last_dice_tokens = [tokens_list[i-1], token]
                    game_history = tokens_list[:i-1]
                    break
                else:
                    # Single dice token (shouldn't happen in proper data, but handle it)
                    last_dice_tokens = [token]
                    game_history = tokens_list[:i]
                    break
            i -= 1

        # If no dice found, use the whole sequence as history and assume current dice
        if not last_dice_tokens:
            game_history = tokens_list
            last_dice_tokens = ['d21']  # Default fallback dice

        # Combine dice tokens for prediction
        dice_roll = ''.join([d[1] for d in last_dice_tokens])  # Convert ['d2','d1'] -> '21'

        print(f"📊 Backed up to last dice: {dice_roll}")
        print(f"🎲 Game history: {' '.join(game_history[-10:])}...")  # Show last 10 tokens

        # Use proper move sequence prediction
        try:
            predictions = predict_backgammon_moves(
                model_single, game_history, dice_roll,
                idx_to_move, move_to_idx, model_device, top_k=5
            )

            # Format predictions for display
            move_predictions = []
            for prob, move_seq in predictions[:5]:
                if move_seq and move_seq[0] != '<NOMOVE>':
                    move_str = ' '.join(move_seq)
                    move_predictions.append(f"{move_str}({prob:.3f})")
                elif move_seq and move_seq[0] == '<NOMOVE>':
                    move_predictions.append(f"NOMOVE({prob:.3f})")

            generated_moves = ' | '.join(move_predictions) if move_predictions else "No valid moves predicted"

            print("\n🎯 Predicted Move Sequences (top-5 with joint probabilities):")
            print(generated_moves)

            all_text = all_text + (f"\nInput Sequence:\n{input_seq_str}\nLast Dice: {dice_roll}\nPredicted Sequences:\n{generated_moves}")

        except Exception as e:
            print(f"⚠️ Prediction failed: {e}")
            # Fallback to simple token prediction
            output, _ = model_single(input_seq)
            logits = output[0, -1]
            probs = torch.softmax(logits, dim=-1)

            idx_sorted = torch.argsort(probs, descending=True)
            all_predictions = []
            for i in idx_sorted[:5].tolist():
                token = idx_to_move.get(i, '<UNK>')
                all_predictions.append((token, probs[i].item()))

            move_predictions = [(tok, conf) for tok, conf in all_predictions if tok.startswith('m_')][:5]
            generated_moves = ' '.join([tok for tok, _ in move_predictions])

            print("\n🔄 Fallback - Generated Moves (filtered, top-5 unique moves):")
            print(generated_moves)

            all_text = all_text + ("\nInput Sequence:\n" + input_seq_str + "\nGenerated Moves:\n" + generated_moves)

    if was_training:
        model.train()

    return all_text


def load_model_file(model_file_path=None):
    """
    Load saved backgammon model checkpoint for inference or training resumption.

    Loads a complete backgammon model checkpoint including architecture, weights,
    training state, and backgammon tokenization. Handles DataParallel compatibility
    and device placement automatically.

    Loading process:
    1. File selection via GUI or direct path
    2. Checkpoint loading with device compatibility
    3. Model architecture reconstruction from saved hyperparameters
    4. Weight loading with DataParallel prefix handling
    5. Backgammon tokenizer restoration
    6. Device placement and optimization

    Returns:
        Tuple for training resumption: (model, vocab_size, n_embd, n_head, n_kv_heads,
                                       block_size, n_layer, dropout, optimizer_state_dict,
                                       scheduler_state_dict, scaler_state_dict, last_epoch,
                                       last_batch_idx, hyperparameters)

    Args:
        model_file_path: Optional direct path to model file (if None, shows file dialog)
    """
    if model_file_path is None:
        print("Please select a backgammon model file.")
        print(f"Opening file dialog in: /home/jonathan/Data")
        model_file = create_file_dialog(title="Select Backgammon Model File", filetypes=[("PyTorch files", "*.pth")], initialdir="/home/jonathan/Data")
    else:
        model_file = model_file_path
        print(f"Loading model file: {model_file}")

    if model_file:
        # Load checkpoint
        if device.type == 'mps':
            checkpoint = torch.load(model_file, map_location='cpu')
        else:
            checkpoint = torch.load(model_file)

        # Get hyperparameters
        hyperparameters = checkpoint['hyperparameters']
        vocab_size = hyperparameters['vocab_size']
        n_embd = hyperparameters['n_embd']
        n_head = hyperparameters['n_head']
        n_layer = hyperparameters['n_layer']
        dropout = hyperparameters['dropout']
        block_size = hyperparameters['block_size']
        n_kv_heads = hyperparameters.get('n_kv_heads', n_head // 3)

        # Load tokenizer
        tokenizer = checkpoint.get('tokenizer')
        if isinstance(tokenizer, dict):
            global move_to_idx, idx_to_move
            move_to_idx = tokenizer
            idx_to_move = {idx: move for move, idx in move_to_idx.items()}
            print(f"Loaded backgammon moves tokenizer with {len(move_to_idx)} tokens")

        # Create backgammon model
        model = BackgammonModel(vocab_size, n_embd, n_head, n_kv_heads, block_size, n_layer, dropout, use_value_head=True)
        model.set_special_ids(move_to_idx)
        # PLAIN: attach a 1/0 mask over vocab so loss applies ONLY to move tokens (m_*)
        try:
            is_move_vec = torch.zeros(vocab_size, dtype=torch.float32)
            for tok, idx in move_to_idx.items():
                # Include explicit moves (m_), No Move, and End of Move marker
                if isinstance(tok, str) and (tok.startswith('m_') or tok in ['<NOMOVE>', '<EOM>']):
                    is_move_vec[idx] = 1.0
            model.register_buffer('is_move_vec', is_move_vec)
            print("✅ Move-token loss mask attached (training ignores dice/special targets)")
        except Exception as e:
            print(f"⚠️ Could not attach move-token mask: {e}")

        # Move model to device before loading state dict
        model = model.to(device)

        # Load state dict with proper handling
        state_dict = checkpoint['model_state_dict']
        cleaned_state_dict = {}
        for key, val in state_dict.items():
            new_key = key
            if new_key.startswith('module.'):
                new_key = new_key[len('module.'):]
            elif new_key.startswith('_orig_mod.module.'):
                new_key = new_key[len('_orig_mod.module.'):]
            elif new_key.startswith('_orig_mod.'):
                new_key = new_key[len('_orig_mod.'):]
            # Skip runtime buffers that we'll recreate
            if new_key in ['is_move_vec', 'move_weights']:
                continue
            cleaned_state_dict[new_key] = val

        model.load_state_dict(cleaned_state_dict, strict=False)

        # Check if checkpoint had frequency weights (accounting for DataParallel prefixes)
        def clean_key(key):
            new_key = key
            if new_key.startswith('module.'):
                new_key = new_key[len('module.'):]
            elif new_key.startswith('_orig_mod.module.'):
                new_key = new_key[len('_orig_mod.module.'):]
            elif new_key.startswith('_orig_mod.'):
                new_key = new_key[len('_orig_mod.'):]
            return new_key

        had_weights = any(clean_key(key) == 'move_weights' for key in state_dict.keys())

        # DEBUG: Show detection results
        weight_keys_found = [key for key in state_dict.keys() if clean_key(key) == 'move_weights']
        print(f"DEBUG: Weight detection - had_weights={had_weights}, keys found: {weight_keys_found}")

        # Always ask user about frequency weighting for consistency
        if had_weights:
            print("⚠️  This checkpoint was trained WITH frequency-based token weighting")
            default_choice = "y"
        else:
            print("✅ This checkpoint was trained WITHOUT frequency-based token weighting")
            default_choice = "n"

        use_weights_input = get_input_with_default(
            "Use frequency-based weighting for rare move tokens? (y/n)", default_choice
        )
        use_weights = use_weights_input.lower().startswith('y')
        print(f"User input: '{use_weights_input}', use_weights={use_weights}")
        print(f"Frequency weighting: {'ENABLED' if use_weights else 'DISABLED'}")

        if use_weights:
            if had_weights:
                # Load existing weights from checkpoint
                try:
                    move_weights_key = next(key for key in state_dict.keys() if clean_key(key) == 'move_weights')
                    print(f"DEBUG: Loading weights from key: {move_weights_key}")
                    weights_tensor = state_dict[move_weights_key]
                    print(f"DEBUG: Weights tensor shape: {weights_tensor.shape}, dtype: {weights_tensor.dtype}")
                    model.register_buffer('move_weights', weights_tensor)
                    print("✅ Frequency weights preserved from checkpoint")
                except Exception as e:
                    print(f"❌ ERROR loading weights from checkpoint: {e}")
                    print("🧹 Falling back to unweighted loss")
            else:
                # Create new weights (this shouldn't happen in inference, but allow it for training)
                print("🔢 Creating frequency-based weights for continued training...")
                # This would require the training text, which we don't have here
                # For now, just note that this path shouldn't be taken in normal usage
                print("⚠️  Creating weights without training text - this may not work properly")
        else:
            # Remove weights if they exist
            if hasattr(model, 'move_weights'):
                print("DEBUG: Removing existing weights from model")
                delattr(model, 'move_weights')
            print("🧹 Using unweighted loss")

        # DEBUG: Final state
        final_has_weights = hasattr(model, 'move_weights') and getattr(model, 'move_weights', None) is not None
        print(f"DEBUG: Model has weights after checkpoint loading: {final_has_weights}")

        # Get optimizer and scheduler states
        optimizer_state_dict = checkpoint.get('optimizer_state_dict')
        scheduler_state_dict = checkpoint.get('scheduler_state_dict')
        scaler_state_dict = checkpoint.get('scaler_state_dict')
        last_epoch = checkpoint.get('epoch', -1)
        last_batch_idx = checkpoint.get('batch_idx', -1)

        print(f"Backgammon model loaded from {model_file}")
        print(f"Model hyperparameters: {hyperparameters}")
        print(f"Last epoch: {last_epoch}, Last batch: {last_batch_idx}")

        return (model, vocab_size, n_embd, n_head, n_kv_heads, block_size, n_layer, dropout,
                optimizer_state_dict, scheduler_state_dict, scaler_state_dict, last_epoch, last_batch_idx, hyperparameters)
    else:
        print("No model file selected.")
        return None


def get_model_module(model):
    """Helper function to safely access the model when it might be wrapped in DataParallel"""
    if isinstance(model, nn.DataParallel) or hasattr(model, 'module'):
        return model.module
    return model


def select_gpus():
    num_gpus = torch.cuda.device_count()
    if num_gpus <= 1:
        print(f"Only {num_gpus} GPU available. Using it for computation.")
        return list(range(num_gpus))

    all_gpus = list(range(num_gpus))
    default_gpus = ",".join(str(i) for i in all_gpus)

    print(f"Available GPUs: {num_gpus}")
    print("Note: Single GPU (0) is most reliable")
    custom_gpus = input(f"Enter GPU indices separated by commas (default: {default_gpus}): ")

    if not custom_gpus.strip():
        print(f"Using all {num_gpus} GPUs")
        return all_gpus

    try:
        gpu_indices = [int(idx.strip()) for idx in custom_gpus.split(',')]
        if all(0 <= idx < num_gpus for idx in gpu_indices):
            if len(gpu_indices) == 1 and gpu_indices[0] == 0:
                print("Selected GPU 0 only - will skip DataParallel for reliable training")
            return gpu_indices
        else:
            print(f"Invalid GPU index. Using all available GPUs.")
            return all_gpus
    except ValueError:
        print(f"Invalid input. Using all available GPUs.")
        return all_gpus


def enter_batch_size(n_embd, n_head, block_size, n_layer, batch_size, gpu_indices):
    """Calculate conservative batch size for stable backgammon training (prevents crashes)"""
    bytes_per_float = 4
    num_gpus = len(gpu_indices)

    # Conservative GPU optimizations for stability
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        print(f"\n🎯 Conservative GPU Optimization for {gpu_name}")
        # Use conservative settings to prevent crashes
        safety_factor = 0.85  # Conservative safety factor for all GPUs
        memory_efficiency = 0.90  # Standard memory efficiency
    else:
        print("\n💻 CPU Mode")
        safety_factor = 0.95  # Very conservative for CPU
        memory_efficiency = 0.85

    if torch.cuda.is_available() and len(gpu_indices) > 0:
        gpu_memory = sum(torch.cuda.get_device_properties(i).total_memory for i in gpu_indices)
    else:
        gpu_memory = 64 * 1024**3  # Default CPU memory estimate

    # Memory calculations optimized for backgammon model
    vocab_size = len(move_to_idx)
    token_embeddings = vocab_size * n_embd * bytes_per_float
    position_embeddings = block_size * n_embd * bytes_per_float

    # Backgammon model uses true GQA (Grouped Query Attention) - calculate correct memory usage
    head_dim = n_embd // n_head
    n_kv_heads = max(1, n_head // 4)  # GQA ratio, but use actual passed value if available

    # Attention weights: Q_proj + K_proj + V_proj + out_proj
    q_proj = n_embd * n_embd  # Q projection: n_embd -> n_embd
    kv_proj = n_embd * n_kv_heads * head_dim * 2  # K,V projections: n_embd -> n_kv_heads * head_dim each
    out_proj = n_embd * n_embd  # Output projection: n_embd -> n_embd
    attention_weights_per_layer = (q_proj + kv_proj + out_proj) * bytes_per_float
    attention_weights = n_layer * attention_weights_per_layer

    feedforward_weights = n_layer * 4 * n_embd * n_embd * bytes_per_float  # SwiGLU (w1,w2,w3)
    rms_norm_weights = n_layer * 2 * n_embd * bytes_per_float  # RMSNorm per block (2 per layer)

    total_model_params = token_embeddings + position_embeddings + attention_weights + feedforward_weights + rms_norm_weights

    optimizer_memory = total_model_params * 2  # Adam optimizer
    gradient_memory = total_model_params

    # Activations per sequence (backgammon model with gradient checkpointing)
    # Gradient checkpointing significantly reduces memory by recomputing forward pass during backprop
    # Only essential activations (embeddings, attention outputs) are stored

    # Input embeddings (token + position) - always stored
    embedding_activations = block_size * n_embd * bytes_per_float * 2

    # Attention computation requires storing Q,K,V for backprop, but with checkpointing this is minimized
    # Approximate attention memory per layer (conservative estimate)
    attention_per_layer = block_size * n_embd * bytes_per_float * 3  # Q,K,V projections
    attention_activations = n_layer * attention_per_layer

    # Feedforward activations - checkpointed, so minimal storage
    ff_per_layer = block_size * n_embd * bytes_per_float * 2  # Input and output of FF
    ff_activations = n_layer * ff_per_layer

    # Gradient checkpointing provides significant memory savings
    # Estimate: 50-70% reduction in activation memory during training
    checkpointing_savings = 0.6  # 60% reduction
    total_per_seq = (embedding_activations + attention_activations + ff_activations) * (1 - checkpointing_savings) * memory_efficiency

    # Adjust for multi-GPU setup
    if num_gpus > 1:
        # DataParallel replicates model on each GPU
        total_model_params *= num_gpus
        optimizer_memory *= num_gpus
        gradient_memory *= num_gpus
        # But activations are split across GPUs
        total_per_seq = total_per_seq / num_gpus

    available_memory = gpu_memory * safety_factor - (total_model_params + optimizer_memory + gradient_memory)
    max_batch_size = max(1, int(available_memory / total_per_seq))

    print(f"\n🧠 Conservative Memory Analysis for {num_gpus} GPU{'s' if num_gpus > 1 else ''} (crash prevention):")
    print(f"- Model parameters: {total_model_params / 1e9:.2f} GB")
    print(f"- Optimizer memory: {optimizer_memory / 1e9:.2f} GB")
    print(f"- Gradient memory: {gradient_memory / 1e9:.2f} GB")
    print(f"- Memory per sequence: {total_per_seq / 1e6:.2f} MB")
    print(f"- Total GPU memory: {gpu_memory / 1e9:.1f} GB")
    print(f"- Available memory: {available_memory / 1e9:.2f} GB")

    if num_gpus == 1:
        print(f"✅ Single GPU - Maximum batch size: {max_batch_size}")

        # Optimized batch size recommendations for GPU training
        if torch.cuda.is_available():
            # GPUs can handle larger batches for backgammon
            recommended_batch = min(max_batch_size, 256)  # Optimized for backgammon performance
        else:
            recommended_batch = min(max_batch_size, 32)  # CPU training

    else:
        print(f"⚠️  Multi-GPU DataParallel - Maximum batch size per GPU: {max_batch_size}")
        print("Note: DataParallel may freeze. Consider using single GPU for better reliability.")

        # Optimized recommendations for multi-GPU setup
        if torch.cuda.is_available():
            # GPUs have good memory for backgammon batches
            recommended_batch = min(max_batch_size, 128)  # Optimized for multi-GPU backgammon performance
        else:
            recommended_batch = min(max_batch_size, 16)  # CPU multi-processing

    batch_size = int(get_input_with_default(f"Enter batch size (recommended: {recommended_batch}, max: {max_batch_size}): ", recommended_batch))
    batch_size = max(1, min(batch_size, max_batch_size))

    return batch_size


# Global variables for backgammon tokenization - will be set when loading data
move_to_idx = None
idx_to_move = None


# Core training function
def _train_backgammon_model_core(text, checkpoint_data=None):
    """
    Core training logic for backgammon move prediction model.

    Handles the complete training pipeline including model setup, GPU configuration,
    optimizer initialization, data loading, and training loop execution. Supports
    both fresh training and checkpoint resumption.

    Training features:
    - GPU optimizations for maximum performance
    - DataParallel support for multi-GPU training
    - Adafactor optimizer for stable backgammon model training
    - Gradient scaling and clipping for training stability
    - Progress monitoring with sample move generation
    - Automatic checkpointing with comprehensive state saving

    Args:
        text: Preprocessed backgammon game text data
        checkpoint_data: Optional tuple from load_model_file() for training resumption
                        Contains: (model, vocab_size, n_embd, n_head, n_kv_heads, block_size,
    """
    print("BackgammonBrain - Backgammon Move Prediction LLM")
    print("=" * 50)
    print(f"DEBUG: Starting _train_backgammon_model_core with text length: {len(text)}")

    # Create vocabulary from the training text
    global move_to_idx, idx_to_move
    
    if checkpoint_data:
        # Resuming from checkpoint: merge checkpoint vocabulary with new tokens from text
        checkpoint_model, checkpoint_vocab_size, n_embd, n_head, n_kv_heads, block_size, n_layer, dropout, \
        optimizer_state_dict, scheduler_state_dict, scaler_state_dict, start_epoch, start_batch, checkpoint_hyperparams = checkpoint_data
        
        if move_to_idx is None:
            print("❌ ERROR: Checkpoint vocabulary not loaded. Cannot merge vocabularies.")
            return
        
        print(f"📚 Merging checkpoint vocabulary ({len(move_to_idx)} tokens) with new dataset tokens...")
        original_vocab_size = len(move_to_idx)
        move_to_idx, new_token_count = merge_vocabularies(move_to_idx, text)
        idx_to_move = create_idx_to_move(move_to_idx)
        
        vocab_size = len(move_to_idx)
        print(f"✅ Merged vocabulary size: {vocab_size}")
        
        # Expand model embeddings if vocabulary grew
        if vocab_size > original_vocab_size:
            expand_model_embeddings(checkpoint_model, original_vocab_size, vocab_size, n_embd)
            
            # Update move token mask for new tokens
            if hasattr(checkpoint_model, 'is_move_vec'):
                for tok, idx in move_to_idx.items():
                    if isinstance(tok, str) and (tok.startswith('m_') or tok in ['<NOMOVE>', '<EOM>']) and idx >= original_vocab_size:
                        checkpoint_model.is_move_vec[idx] = 1.0
                print(f"✅ Updated move token mask for {new_token_count} new tokens")
            
            # Note: Optimizer state for embedding and lm_head layers will be invalidated
            # We need to filter out optimizer state for parameters that changed size
            # This prevents RuntimeError when optimizer tries to update mismatched parameter shapes
            print(f"ℹ️  Note: Optimizer state for new token embeddings will be reinitialized")
            print(f"   This is expected - new tokens start with fresh optimizer state")
            
            # Mark that we need to filter optimizer state and save sizes for logging
            vocab_expanded = True
            vocab_old_size = original_vocab_size
            vocab_new_size = vocab_size
        else:
            vocab_expanded = False
            vocab_old_size = None
            vocab_new_size = None
    else:
        # Fresh start: create vocabulary from scratch
        if move_to_idx is None:
            move_to_idx = create_move_to_idx_from_text(text)
            idx_to_move = create_idx_to_move(move_to_idx)
        vocab_expanded = False
        vocab_old_size = None
        vocab_new_size = None

    vocab_size = len(move_to_idx)
    print(f"Final vocabulary size: {vocab_size}")

    # Ensure gpu_indices is accessible (capture from global scope)
    global gpu_indices
    if 'gpu_indices' not in globals() or gpu_indices is None:
        gpu_indices = [0] if torch.cuda.is_available() else None

    # Set default values for all variables that might be needed
    learning_rate = BACKGAMMON_DEFAULTS['learning_rate']
    weight_decay = BACKGAMMON_DEFAULTS['weight_decay']

    # Check if resuming from checkpoint
    if checkpoint_data:
        # Use the already unpacked checkpoint_model (renamed to model for consistency)
        model = checkpoint_model
        # Update vocab_size to reflect merged vocabulary (may have grown)
        vocab_size = len(move_to_idx)
        model.set_special_ids(move_to_idx)
        print(f"Resuming from checkpoint: epoch {start_epoch}, batch {start_batch}")

        # Display model architecture (cannot be changed)
        print(f"Model architecture: {n_layer} layers, {n_head} heads, {n_embd} embedding dim, dropout {dropout}")
        print(f"Block size: {block_size}, Vocab size: {vocab_size} (after merge)")

        # Display training setup from checkpoint
        saved_gpu_indices = checkpoint_hyperparams.get('gpu_indices')
        # Capture early GPU selection before any assignments (access global to avoid UnboundLocalError)
        early_gpu_selection = globals().get('gpu_indices')
        if saved_gpu_indices:
            print(f"📋 Checkpoint originally trained on GPU{'s' if len(saved_gpu_indices) > 1 else ''}: {saved_gpu_indices}")
            print(f"   System has {torch.cuda.device_count()} GPUs available: {[f'GPU{i}' for i in range(torch.cuda.device_count())]}")

            # For single GPU, give user choice to use different GPU
            if len(saved_gpu_indices) == 1:
                gpu_input = get_input_with_default(
                    f"Resume on GPU {saved_gpu_indices[0]} or enter new GPU number", str(saved_gpu_indices[0])
                )

                try:
                    new_gpu = int(gpu_input)
                    # For single GPU resumption, trust the user's input since early selection already validated
                    gpu_indices = [new_gpu]
                    print(f"Selected GPU: {new_gpu}")
                    os.environ['CUDA_VISIBLE_DEVICES'] = str(new_gpu)
                    print(f"CUDA_VISIBLE_DEVICES updated to: {new_gpu}")
                except ValueError:
                    print(f"Invalid input. Using current GPU {early_gpu_selection[0] if early_gpu_selection else saved_gpu_indices[0]}")
                    gpu_indices = early_gpu_selection if early_gpu_selection else saved_gpu_indices
            else:
                # Multi-GPU: force same GPUs for state consistency
                print(f"🔄 Auto-selecting same GPUs for resume: {saved_gpu_indices}")
                gpu_indices = saved_gpu_indices
                # Update CUDA_VISIBLE_DEVICES for checkpoint GPUs
                os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, gpu_indices))
                print(f"CUDA_VISIBLE_DEVICES updated to: {os.environ['CUDA_VISIBLE_DEVICES']}")
                print(f"⚠️  IMPORTANT: Must use SAME number of GPUs as original training")
                print(f"   Checkpoint has states for {len(saved_gpu_indices)} GPUs - cannot change GPU count")
                print(f"   1-GPU checkpoint → load on 1-GPU ✅ | 3-GPU checkpoint → load on 3-GPUs ✅")
                print(f"   1-GPU checkpoint → load on 3-GPUs ❌ | 3-GPU checkpoint → load on 1-GPU ❌")
        else:
            print("GPU information not available in checkpoint")
            # Keep existing gpu_indices from early selection

        # Ensure gpu_indices is always set after checkpoint loading logic
        # (but don't override if already set by checkpoint)
        try:
            if gpu_indices is None:
                # Fallback: this shouldn't happen but prevents UnboundLocalError
                gpu_indices = [0] if torch.cuda.is_available() else None
        except NameError:
            gpu_indices = [0] if torch.cuda.is_available() else None

        # Ask for training parameters (use checkpoint values as defaults where available)
        # These can be safely changed without breaking the model
        saved_batch_size = checkpoint_hyperparams.get('batch_size', 256)
        print(f"Note: Changing batch size may affect training resumption accuracy")
        batch_size = int(get_input_with_default("Batch size", saved_batch_size))
        num_epochs = int(get_input_with_default("Number of epochs", 20))

        # Learning rate and weight decay can be adjusted (will be loaded from optimizer state)
        saved_learning_rate = checkpoint_hyperparams.get('learning_rate', BACKGAMMON_DEFAULTS['learning_rate'])
        print(f"Current learning rate from checkpoint: {saved_learning_rate}")
        learning_rate_input = get_input_with_default("Learning rate (or press Enter to keep current)", saved_learning_rate)
        learning_rate = float(learning_rate_input)

        saved_weight_decay = checkpoint_hyperparams.get('weight_decay', BACKGAMMON_DEFAULTS['weight_decay'])
        weight_decay_input = get_input_with_default("Weight decay", saved_weight_decay)
        weight_decay = float(weight_decay_input)

        saved_dropout = checkpoint_hyperparams['dropout']
        dropout_input = get_input_with_default("Dropout", saved_dropout)
        dropout = float(dropout_input)
    else:
        # Fresh start - use defaults
        start_epoch = 0
        start_batch = 0
        optimizer_state_dict = None
        scheduler_state_dict = None
        scaler_state_dict = None

        # Single GPU mode - use interactive prompts or defaults
        n_embd = BACKGAMMON_DEFAULTS['n_embd']
        n_head = BACKGAMMON_DEFAULTS['n_head']
        n_kv_heads = BACKGAMMON_DEFAULTS['n_kv_heads']
        block_size = BACKGAMMON_DEFAULTS['block_size']
        n_layer = BACKGAMMON_DEFAULTS['n_layer']
        dropout = BACKGAMMON_DEFAULTS['dropout']
        batch_size = BACKGAMMON_DEFAULTS['batch_size']
        num_epochs = BACKGAMMON_DEFAULTS['num_epochs']

        # Allow parameter overrides
        n_embd = int(get_input_with_default("Embedding dimensions", n_embd))
        n_head = int(get_input_with_default("Number of query heads", n_head))
        n_kv_heads = int(get_input_with_default("Number of KV heads", n_kv_heads))
        block_size = int(get_input_with_default("Sequence length", block_size))
        n_layer = int(get_input_with_default("Number of layers", n_layer))
        dropout = float(get_input_with_default("Dropout", dropout))
        batch_size = int(get_input_with_default("Batch size", batch_size))
        num_epochs = int(get_input_with_default("Number of epochs", num_epochs))

        # Create your own BackgammonModel for backgammon move prediction
        model = BackgammonModel(vocab_size, n_embd, n_head, n_kv_heads, block_size, n_layer, dropout, use_chess=True, use_value_head=True)
        model.set_special_ids(move_to_idx)
      # PLAIN: attach move-only loss mask for fresh training too
        try:
            is_move_vec = torch.zeros(vocab_size, dtype=torch.float32)
            for tok, idx in move_to_idx.items():
                if isinstance(tok, str) and (tok.startswith('m_') or tok in ['<NOMOVE>', '<EOM>']):
                    is_move_vec[idx] = 1.0
            model.register_buffer('is_move_vec', is_move_vec)
            print("✅ Move-token loss mask attached (training ignores dice/special targets)")
        except Exception as e:
            print(f"⚠️ Could not attach move-token mask: {e}")

        print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")

    # Choose optimizer and scheduler early - needed for both single and multi-GPU setups
    clip_threshold = BACKGAMMON_DEFAULTS['max_norm']
    # Only ask for optimizer choice if not loading from checkpoint
    if not checkpoint_data:
        optimizer_choice = get_input_with_default("Optimizer (adamw/adfactor)", "adamw").lower()
        scheduler_choice = get_input_with_default("Scheduler (cosine/plateau) [plateau=recommended for stuck training]", "plateau").lower()
    else:
        # When loading checkpoint, ask if they want to switch schedulers
        print("\n📋 Current checkpoint was saved with CosineAnnealingLR scheduler")
        print("   This scheduler decays learning rate regardless of training progress")
        print("   Consider switching to ReduceLROnPlateau for better plateau handling")
        scheduler_choice = get_input_with_default("Keep current scheduler or switch? (cosine/plateau)", "plateau").lower()
        optimizer_choice = "adamw"  # Default, will be overridden by loaded state

    # Configure GPU training with optimizations for backgammon model
    if torch.cuda.is_available():
        # For checkpoint loading, gpu_indices is already set by checkpoint logic above
        # For fresh training, gpu_indices needs to be set
        if checkpoint_data is None:
            # Fresh training - need to select GPUs if not already set
            if gpu_indices is None:
                gpu_indices = select_gpus()
        # else: checkpoint_data exists, gpu_indices already set by checkpoint loading logic above

        if gpu_indices and len(gpu_indices) > 0:
            print(f"\n🚀 Setting up backgammon training on GPU{'s' if len(gpu_indices) > 1 else ''}: {gpu_indices}")
            device = torch.device('cuda')

            # Always move model to device first
            model = model.to(device)

            # Initialize multi-GPU flag (only if not already set by checkpoint loading)
            if 'use_custom_parallel' not in locals():
                use_custom_parallel = False

            # Multi-GPU setup - Custom Parallel Training (NOT DataParallel)
            # =================================================================
            # Why Custom Multi-GPU instead of DataParallel:
            # - DataParallel has GIL bottlenecks, CUDA sync issues, memory replication
            # - Custom approach: Each GPU gets its own model/optimizer/scheduler/scaler
            # - Manual gradient averaging avoids DataParallel's complex synchronization
            # - No Python multiprocessing = no GIL issues
            # - Each GPU processes independent batch chunks in parallel
            #
            # Architecture: N GPUs = N identical models, N optimizers, N schedulers, N scalers
            # Training: Split batch across GPUs → forward/backward independently → average gradients → update all models
            # =================================================================
            if len(gpu_indices) > 1:
                print(f"🚀 MULTI-GPU MODE: Custom parallel training across {len(gpu_indices)} GPUs")
                print(f"   Each GPU gets independent model/optimizer/scheduler/scaler replica")
                print(f"   Manual gradient averaging avoids DataParallel GIL/sync issues")
                if checkpoint_data:
                    print(f"   🔄 Resuming multi-GPU training from checkpoint")

            if len(gpu_indices) > 1:  # After checkpoint check
                print(f"🔄 Attempting to setup multi-GPU training on GPUs: {gpu_indices}")
                models = []  # Initialize empty list for safety
                try:
                    # Validate that requested GPUs are available
                    available_gpus = torch.cuda.device_count()
                    max_requested_gpu = max(gpu_indices)
                    print(f"   Requested max GPU index: {max_requested_gpu}, System has {available_gpus} GPUs (indices 0-{available_gpus-1})")

                    if max_requested_gpu >= available_gpus:
                        print(f"⚠️  WARNING: Checkpoint trained on GPUs {gpu_indices} but only {available_gpus} GPUs available")
                        print(f"   GPU {max_requested_gpu} not available on this system")
                        print(f"   Falling back to single GPU mode on GPU 0")
                        gpu_indices = [0]
                        use_custom_parallel = False
                    else:
                        # Create model replicas manually on each GPU
                        models = []
                        optimizers = []
                        schedulers = []
                        scalers = []

                        for i, gpu_idx in enumerate(gpu_indices):
                            # Create model on specific GPU
                            model_gpu = BackgammonModel(vocab_size, n_embd, n_head, n_kv_heads, block_size, n_layer, dropout, use_chess=True, use_value_head=True)
                            model_gpu.set_special_ids(move_to_idx)

                            # If loading from checkpoint, copy the loaded weights to each GPU model
                            if checkpoint_data:
                                # Clean state_dict to skip runtime buffers like is_move_vec
                                gpu_state_dict = {}
                                for key, val in model.state_dict().items():
                                    if key not in ['is_move_vec', 'move_weights']:
                                        gpu_state_dict[key] = val
                                model_gpu.load_state_dict(gpu_state_dict)

                            # Register move token mask buffer on GPU model
                            try:
                                is_move_vec = torch.zeros(vocab_size, dtype=torch.float32)
                                for tok, idx in move_to_idx.items():
                                    if isinstance(tok, str) and (tok.startswith('m_') or tok in ['<NOMOVE>', '<EOM>']):
                                        is_move_vec[idx] = 1.0
                                model_gpu.register_buffer('is_move_vec', is_move_vec)
                            except Exception as e:
                                print(f"⚠️ Could not attach move-token mask to GPU {i}: {e}")

                            model_gpu = model_gpu.to(f'cuda:{i}')  # Use remapped indices

                            # Create optimizer for this GPU
                            if optimizer_choice == 'adamw':
                                opt = torch.optim.AdamW(
                                    model_gpu.parameters(),
                                    lr=learning_rate,
                                    weight_decay=weight_decay,
                                    betas=(0.9, 0.999),
                                    eps=1e-8
                                )
                            else:
                                opt = Adafactor(
                                    model_gpu.parameters(),
                                    lr=learning_rate,
                                    scale_parameter=True,
                                    relative_step=False,
                                    warmup_init=False,
                                    clip_threshold=clip_threshold,
                                    weight_decay=weight_decay,
                                    beta1=0.9,
                                    eps=(1e-30, 1e-3)
                                )
                            optimizers.append(opt)

                            # Create scheduler for this GPU
                            if scheduler_choice == 'plateau':
                                sched = ReduceLROnPlateau(opt, mode='min', factor=0.8, patience=100, threshold=0.001, min_lr=1e-7)
                            else:
                                sched = CosineAnnealingLR(opt, T_max=num_epochs)
                            schedulers.append(sched)

                            # Create scaler for this GPU
                            scal = GradScaler()
                            scalers.append(scal)

                            # Add the model to the list (this was missing!)
                            models.append(model_gpu)

                        print(f"✅ Custom multi-GPU setup complete: {len(models)} models created on GPUs: {gpu_indices}")
                        use_custom_parallel = True

                except Exception as e:
                    print(f"⚠️  ERROR: Failed to setup multi-GPU training: {e}")
                    print(f"   Falling back to single GPU mode on GPU 0")
                    gpu_indices = [0]
                    use_custom_parallel = False

            # Load checkpoint states for multi-GPU resume (only if multi-GPU setup succeeded)
            if checkpoint_data and use_custom_parallel:
                optimizer_state_dict = checkpoint_data[8]  # optimizer states (list for multi-GPU)
                scheduler_state_dict = checkpoint_data[9]  # scheduler states (list for multi-GPU)
                scaler_state_dict = checkpoint_data[10]    # scaler states (list for multi-GPU)

                if isinstance(optimizer_state_dict, list) and len(optimizer_state_dict) == len(optimizers):
                    if not optimizer_state_matches(optimizers[0], optimizer_state_dict):
                        pass
                    else:
                        print(f"Loading {len(optimizer_state_dict)} optimizer states for {len(optimizers)} GPUs")
                        # When vocabulary expands, we must filter out old optimizer state for changed parameters
                        if vocab_expanded:
                            print(f"🔧 Filtering optimizer state for vocabulary expansion ({vocab_old_size}→{vocab_new_size} tokens)...")
                            optimizer_state_dict = filter_optimizer_state(optimizer_state_dict, models[0])
                        for i, opt_state in enumerate(optimizer_state_dict):
                            optimizers[i].load_state_dict(opt_state)
                            print(f"  GPU {gpu_indices[i]}: optimizer state loaded")

                            # Always reset LR when switching to cosine scheduler (multi-GPU)
                            if scheduler_choice == 'cosine':
                                optimizers[i].param_groups[0]['lr'] = learning_rate
                                print(f"  GPU {gpu_indices[i]}: reset LR to {learning_rate} for cosine scheduler")

                # Always start with fresh scheduler states when loading checkpoint
                print(f"🔄 Using fresh {scheduler_choice} schedulers for all GPUs (allows switching types for stuck runs)")

                if isinstance(scaler_state_dict, list) and len(scaler_state_dict) == len(scalers):
                    print(f"Loading {len(scaler_state_dict)} scaler states for {len(scalers)} GPUs")
                    for i, scal_state in enumerate(scaler_state_dict):
                        scalers[i].load_state_dict(scal_state)

                print("✅ Multi-GPU checkpoint states loaded - training resumes with all states preserved")
            # Only fall back to single GPU if we're not in multi-GPU mode
            if not use_custom_parallel:
                # Single GPU mode - optimal for backgammon training stability
                print(f"✅ SINGLE GPU MODE: Using GPU {gpu_indices[0]} (recommended for backgammon)")
                print(f"   Optimized for backgammon move prediction training")
                # Model already moved to device above
                use_custom_parallel = False

                # Same as Chess on this Spark: compile the model so the GPU runs fused kernels.
                # mode='default' on purpose. reduce-overhead keeps CUDA graphs that can freeze a GB10.
                if not os.environ.get('BACKGAMMON_NO_COMPILE') and hasattr(torch, 'compile'):
                    try:
                        print("Enabling torch.compile()...")
                        _eager_model = model
                        model = torch.compile(model, mode='default')
                        with torch.no_grad():
                            _warm_t = min(int(getattr(_eager_model, 'block_size', 128)), 256)
                            _warm = torch.full((2, _warm_t), move_to_idx['<STARTGAME>'], dtype=torch.long, device=device)
                            model(_warm)
                        print("torch.compile() enabled")
                    except Exception as e:
                        print(f"torch.compile() failed: {str(e).strip().splitlines()[-1][:200]}")
                        print("Continuing without torch.compile (set BACKGAMMON_NO_COMPILE=1 to skip the attempt)")
                        model = _eager_model
                else:
                    print("torch.compile() skipped")
        else:
            print("❌ ERROR: No CUDA GPUs available. This backgammon training requires GPU support.")
            print("   Please run on a system with GPU support (CUDA or MPS).")
            exit(1)
    else:
        # Handle MPS (Mac) - GPU only, no CPU fallback
        if torch.backends.mps.is_available():
            print("🚀 Setting up backgammon training on MPS GPU")
            device = torch.device('mps')
            gpu_indices = []  # MPS doesn't use gpu_indices like CUDA
            model = model.to(device)
            use_custom_parallel = False  # MPS doesn't use custom parallel training
        else:
            print("❌ ERROR: No GPU available. This backgammon training requires MPS (Mac) or CUDA GPU.")
            print("   Please run on a system with GPU support.")
            exit(1)

    # Create optimizer/scheduler/scaler - only for single GPU mode
    if not use_custom_parallel:
        model_params = get_model_module(model).parameters()

        if optimizer_choice == 'adamw':
            print("🚀 Using AdamW optimizer (better convergence for backgammon models)")
            optimizer = torch.optim.AdamW(
                model_params,
                lr=learning_rate,
                weight_decay=weight_decay,
                betas=(0.9, 0.999),
                eps=1e-8
            )
        else:
            print("🔧 Using Adafactor optimizer (memory efficient for large models)")
            optimizer = Adafactor(
                model_params,
                lr=learning_rate,
                scale_parameter=True,
                relative_step=False,
                warmup_init=False,
                clip_threshold=clip_threshold,
                weight_decay=weight_decay,
                beta1=0.9,
                eps=(1e-30, 1e-3)
            )

        # Setup scheduler and scaler
        if scheduler_choice == 'plateau':
            scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=100, threshold=0.001, min_lr=1e-7)
        else:
            scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)
        scaler = GradScaler()

        # Restore optimizer and scheduler state if resuming from checkpoint
        if checkpoint_data and optimizer_state_dict:
            if not optimizer_state_matches(optimizer, optimizer_state_dict):
                pass
            else:
                print("Loading optimizer state from checkpoint")
                # When vocabulary expands, we must filter out old optimizer state for changed parameters
                if vocab_expanded:
                    print(f"🔧 Filtering optimizer state for vocabulary expansion ({vocab_old_size}→{vocab_new_size} tokens)...")
                    optimizer_state_dict = filter_optimizer_state(optimizer_state_dict, model)
                # Handle case where checkpoint has multi-GPU states but we're loading in single GPU mode
                if isinstance(optimizer_state_dict, list):
                    print("⚠️  Checkpoint has multi-GPU optimizer states, using first one for single GPU resume")
                    optimizer.load_state_dict(optimizer_state_dict[0])
                else:
                    optimizer.load_state_dict(optimizer_state_dict)

                # Always reset LR when switching to cosine scheduler (it should start fresh, not inherit plateau's crushed LR)
                if scheduler_choice == 'cosine':
                    print(f"Switching to cosine scheduler - resetting LR to {learning_rate} (ignoring checkpoint LR)")
                    optimizer.param_groups[0]['lr'] = learning_rate
                    print(f"Cosine scheduler starting with fresh learning rate: {learning_rate}")
                else:
                    # For plateau scheduler, keep the checkpoint LR (it manages its own decay)
                    current_lr = optimizer.param_groups[0]['lr']
                    learning_rate = current_lr  # Update our variable to match
                    print(f"Resumed with learning rate: {learning_rate}")

        # Always start with fresh scheduler state when loading checkpoint
        print(f"🔄 Using fresh {scheduler_choice} scheduler (allows switching types for stuck runs)")

        if checkpoint_data and scaler_state_dict:
            print("Loading scaler state from checkpoint")
            # Handle case where checkpoint has multi-GPU states but we're loading in single GPU mode
            if isinstance(scaler_state_dict, list):
                print("⚠️  Checkpoint has multi-GPU scaler states, using first one for single GPU resume")
                scaler.load_state_dict(scaler_state_dict[0])
            else:
                scaler.load_state_dict(scaler_state_dict)

    # Create dataset and dataloader
    dataset = BackgammonMovesDataset(text, block_size, move_to_idx)
    print(f"Dataset size: {len(dataset)} sequences")

    # Check GPU capability
    gpu_capability = torch.cuda.get_device_capability(0) if torch.cuda.is_available() else (8, 6)

    # Configure data loading for optimal backgammon training performance
    sampler = None
    if isinstance(model, nn.DataParallel):
        # Multi-GPU DataParallel setup - use same config as single GPU for stability
        print(f"DataParallel: Using stable configuration across {len(model.device_ids)} GPUs")
        num_workers = 0  # Same as single GPU - prevents hanging in DataParallel
        pin_memory = device.type == 'cuda'  # Pin memory for faster GPU transfers
        persistent_workers = False  # Disable for stability
        prefetch_factor = None  # Disable prefetch
    else:
        # Same loader settings as Chess. On the Spark, pin_memory copies the same RAM twice, so it stays off.
        if _uses_unified_memory():
            num_workers = min(2, os.cpu_count() or 1)
            pin_memory = False
        else:
            num_workers = min(8, (os.cpu_count() or 2) // 2)
            pin_memory = device.type == 'cuda'
        persistent_workers = num_workers > 0
        prefetch_factor = 2 if num_workers > 0 else None
        print(f"Single GPU data loader: {num_workers} workers, pin_memory={pin_memory}")

    # Print the number of model parameters
    num_params = sum(p.numel() for p in get_model_module(model).parameters())
    print(f"Number of model parameters: {num_params}")
    if isinstance(model, nn.DataParallel):
        print(f"Model's primary device: {next(get_model_module(model).parameters()).device}")
        print(f"Model distributed across devices: {model.device_ids}")
    else:
        print(f"Model is on device: {next(model.parameters()).device}")

    print(f"Using {num_workers} workers for data loading")

    # Memory usage print
    if torch.cuda.is_available():
        if len(gpu_indices) > 0:
            for i, physical_gpu in zip(range(len(gpu_indices)), gpu_indices):
                print(f"GPU {physical_gpu} (device {i}) memory allocated: {torch.cuda.memory_allocated(i) / 1e9:.2f} GB")
                print(f"GPU {physical_gpu} (device {i}) memory reserved: {torch.cuda.memory_reserved(i) / 1e9:.2f} GB")

    # Training loop - handles both single GPU and custom multi-GPU
    if use_custom_parallel and models is not None and len(models) > 0:
        # Custom multi-GPU training loop
        print(f"🎯 Using custom multi-GPU training - {len(models)} GPUs work simultaneously!")
        print(f"   Expected speedup: ~{len(models)}x faster than single GPU (minus overhead)")
        running_loss = 0.0
        total_batches = 0
        epoch_losses = []
        all_text = ""
        inference_frequency = 100

        # Set all models to training mode
        for model_gpu in models:
            model_gpu.train()

        for epoch in range(start_epoch, num_epochs):
            epoch_loss = 0.0
            epoch_batches = 0

            # Create dataloader for this epoch
            data_loader = DataLoader(dataset, batch_size=batch_size,
                                    sampler=sampler, shuffle=(sampler is None),
                                    drop_last=True, num_workers=num_workers,
                                    pin_memory=pin_memory, persistent_workers=persistent_workers,
                                    prefetch_factor=prefetch_factor)

            print(f"DataLoader length: {len(data_loader)}, Epoch: {epoch+1}/{num_epochs}")

            for batch_idx, (x, y, token_weights) in enumerate(data_loader):
                if epoch == start_epoch and batch_idx < start_batch:
                    continue

                # Split batch across GPUs
                num_gpus = len(models)
                batch_size_per_gpu = x.shape[0] // num_gpus
                x_splits = torch.split(x, batch_size_per_gpu)
                y_splits = torch.split(y, batch_size_per_gpu)
                w_splits = torch.split(token_weights, batch_size_per_gpu)

                # Accumulators for correct loss averaging
                batch_weighted_loss_sum = 0.0
                batch_total_tokens = 0

                # Forward and backward pass on each GPU independently
                for gpu_idx, (model_gpu, opt_gpu, scal_gpu, x_gpu, y_gpu, w_gpu) in enumerate(zip(models, optimizers, scalers, x_splits, y_splits, w_splits)):
                    x_gpu = x_gpu.to(f'cuda:{gpu_idx}', non_blocking=True)
                    y_gpu = y_gpu.to(f'cuda:{gpu_idx}', non_blocking=True)
                    w_gpu = w_gpu.to(f'cuda:{gpu_idx}', non_blocking=True)

                    # Forward pass with mixed precision
                    with autocast(device_type='cuda', dtype=torch.float16, enabled=True):
                        # Unpack (loss, num_tokens) tuple
                        output, (loss, num_tokens) = model_gpu(x_gpu, targets=y_gpu, token_weights=w_gpu)

                    # Backward pass
                    opt_gpu.zero_grad(set_to_none=True)
                    scal_gpu.scale(loss).backward()

                    # Unscale gradients
                    scal_gpu.unscale_(opt_gpu)

                    # Clip gradients
                    torch.nn.utils.clip_grad_norm_(model_gpu.parameters(), clip_threshold)

                    # Accumulate weighted loss for correct reporting.
                    # Forward returns a device count so compile stays one graph; read it here, outside compile.
                    if torch.is_tensor(num_tokens):
                        num_tokens = int(num_tokens.item())
                    batch_weighted_loss_sum += loss.item() * num_tokens
                    batch_total_tokens += num_tokens

                # Average gradients across all models
                for model_gpu in models:
                    for param in model_gpu.parameters():
                        if param.grad is not None:
                            param.grad.data /= num_gpus

                # Update all models with averaged gradients
                for opt_gpu, scal_gpu in zip(optimizers, scalers):
                    scal_gpu.step(opt_gpu)
                    scal_gpu.update()

                # Correct weighted average loss across all GPUs
                if batch_total_tokens > 0:
                    avg_loss = batch_weighted_loss_sum / batch_total_tokens
                else:
                    avg_loss = 0.0
                
                running_loss += avg_loss
                epoch_loss += avg_loss
                total_batches += 1
                epoch_batches += 1

                # Progress reporting
                if (batch_idx + 1) % inference_frequency == 0:
                    avg_running_loss = running_loss / total_batches
                    print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(data_loader)}], Avg Loss: {avg_running_loss:.4f}")

                    # Continuous plateau detection
                    if scheduler_choice == 'plateau':
                        if 'batch_group_losses' not in locals():
                            batch_group_losses = []
                            plateau_check_counter = 0
                        batch_group_losses.append(avg_running_loss)
                        plateau_check_counter += 1

                        if len(batch_group_losses) > 20:
                            batch_group_losses = batch_group_losses[-20:]

                        if len(batch_group_losses) >= 20 and plateau_check_counter % 20 == 0:
                            current_2000_batch_avg = sum(batch_group_losses) / len(batch_group_losses)

                            if 'previous_2000_batch_avg' in locals():
                                improvement = previous_2000_batch_avg - current_2000_batch_avg
                                if improvement >= 0.001:
                                    print(f"📈 Improvement: {improvement:.4f} (>= 0.001) - continuing")
                                else:
                                    print(f"🔻 Plateau: {improvement:.4f} < 0.001 - reducing LR")
                                    for sched in schedulers:
                                        sched.step(float('inf'))

                            previous_2000_batch_avg = current_2000_batch_avg

                    # Generate sample using first model
                    all_text = test_progress(
                        epoch, num_epochs, batch_idx, data_loader, avg_loss,
                        models[0], x_splits[0].to(f'cuda:0'), 50, all_text, idx_to_move
                    )

                    # Save using first model and ALL optimizer/scheduler states
                    all_optimizer_states = [opt.state_dict() for opt in optimizers]
                    all_scheduler_states = [sched.state_dict() for sched in schedulers]
                    all_scaler_states = [scal.state_dict() for scal in scalers]

                    save_model_all(
                        models[0], all_text, n_embd, n_head, n_kv_heads, n_layer, dropout,
                        block_size, epoch, batch_idx, batch_size, all_optimizer_states, all_scheduler_states, all_scaler_states, avg_loss,
                        learning_rate, weight_decay, gpu_indices
                    )

                    running_loss = 0.0
                    total_batches = 0

                    # Clear cache on all GPUs
                    for gpu_idx in range(num_gpus):
                        torch.cuda.set_device(gpu_idx)
                        torch.cuda.empty_cache()

            # Epoch summary
            avg_epoch_loss = epoch_loss / epoch_batches if epoch_batches > 0 else 0
            epoch_losses.append(avg_epoch_loss)
            print(f"Epoch [{epoch+1}/{num_epochs}] completed. Average Loss: {avg_epoch_loss:.4f}")

            # Generate sample moves at end of each epoch to show learning progress
            try:
                # Check if vocabulary is available
                if move_to_idx is None or idx_to_move is None:
                    print(f"⚠️  Multi-GPU Epoch {epoch+1}: Vocabulary not ready yet")
                    continue

                # Create a simple test sequence for inference
                start_token = move_to_idx['<STARTGAME>']
                sample_tokens = [start_token] + [move_to_idx.get('41lpab', 0)] * 10  # Simple test sequence
                sample_x = torch.tensor([sample_tokens], dtype=torch.long).to(f'cuda:0')

                epoch_all_text = test_progress(
                    epoch, num_epochs, batch_idx, data_loader, avg_epoch_loss,
                    models[0], sample_x, 30, "", idx_to_move
                )
                print(f"🎯 Multi-GPU Epoch {epoch+1} inference completed")
            except Exception as e:
                print(f"⚠️  Multi-GPU Epoch {epoch+1} inference failed: {e}")
                import traceback
                traceback.print_exc()

    else:
        # Single GPU training loop
        model.train()
        running_loss = 0.0
        total_batches = 0
        epoch_losses = []
        all_text = ""
        inference_frequency = 100

        for epoch in range(start_epoch, num_epochs):
            epoch_loss = 0.0
            epoch_batches = 0

            # Create dataloader for each epoch
            data_loader = DataLoader(dataset, batch_size=batch_size,
                                    sampler=sampler, shuffle=(sampler is None),
                                    drop_last=True, num_workers=num_workers,
                                    pin_memory=pin_memory, persistent_workers=persistent_workers,
                                    prefetch_factor=prefetch_factor)

            print(f"DataLoader length: {len(data_loader)}, Epoch: {epoch+1}/{num_epochs}")

            for batch_idx, (x, y, token_weights) in enumerate(data_loader):
                if epoch == start_epoch and batch_idx < start_batch:
                    continue

                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                token_weights = token_weights.to(device, non_blocking=True)

                if batch_idx == 0:
                    print(f"Batch shapes: x={x.shape}, y={y.shape}")

                # Forward pass with mixed precision
                if device.type == 'cuda':
                    with autocast(device_type='cuda', dtype=torch.float16, enabled=True):
                        # Unpack (loss, num_tokens) tuple
                        output, (loss, num_tokens) = model(x, targets=y, token_weights=token_weights)

                    optimizer.zero_grad(set_to_none=True)
                    scaler.scale(loss).backward()

                    scaler.unscale_(optimizer)
                    model_params = get_model_module(model).parameters()
                    total_norm = torch.nn.utils.clip_grad_norm_(list(model_params), clip_threshold)

                    scaler.step(optimizer)
                    scaler.update()
                else:
                    # Unpack (loss, num_tokens) tuple
                    output, (loss, num_tokens) = model(x, targets=y, token_weights=token_weights)
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()

                loss_val = loss.item()
                running_loss += loss_val
                epoch_loss += loss_val
                total_batches += 1
                epoch_batches += 1

                # Memory management
                current_loss_value = loss_val
                current_batch_x = x

                # Explicit cleanup
                del output, loss, x, y, token_weights
                if device.type == 'cuda':
                    torch.cuda.empty_cache()

                # Progress reporting
                if (batch_idx + 1) % inference_frequency == 0:
                    avg_loss = running_loss / total_batches
                    print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(data_loader)}], Avg Loss: {avg_loss:.4f}")

                    # Continuous plateau detection
                    if scheduler_choice == 'plateau':
                        if 'batch_group_losses' not in locals():
                            batch_group_losses = []
                            plateau_check_counter = 0
                        batch_group_losses.append(avg_loss)
                        plateau_check_counter += 1

                        if len(batch_group_losses) > 20:
                            batch_group_losses = batch_group_losses[-20:]

                        if len(batch_group_losses) >= 20 and plateau_check_counter % 20 == 0:
                            current_2000_batch_avg = sum(batch_group_losses) / len(batch_group_losses)

                            if 'previous_2000_batch_avg' in locals():
                                improvement = previous_2000_batch_avg - current_2000_batch_avg
                                if improvement >= 0.001:
                                    print(f"📈 Improvement: {improvement:.4f} (>= 0.001) - continuing")
                                else:
                                    print(f"🔻 Plateau: {improvement:.4f} < 0.001 - reducing LR")
                                    scheduler.step(float('inf'))

                            previous_2000_batch_avg = current_2000_batch_avg

                    running_loss = 0.0
                    total_batches = 0

                    # Generate sample moves
                    all_text = test_progress(
                        epoch, num_epochs, batch_idx, data_loader, current_loss_value,
                        model, current_batch_x, 50, all_text, idx_to_move
                    )

                    # Save model
                    save_model_all(
                        model, all_text, n_embd, n_head, n_kv_heads, n_layer, dropout,
                        block_size, epoch, batch_idx, batch_size, optimizer, scheduler, scaler, current_loss_value,
                        learning_rate, weight_decay, gpu_indices
                    )

            # Epoch summary
            avg_epoch_loss = epoch_loss / epoch_batches if epoch_batches > 0 else 0
            epoch_losses.append(avg_epoch_loss)
            print(f"Epoch [{epoch+1}/{num_epochs}] completed. Average Loss: {avg_epoch_loss:.4f}")

            # Generate sample moves at end of each epoch to show learning progress
            try:
                # Check if vocabulary is available
                if move_to_idx is None or idx_to_move is None:
                    print(f"⚠️  Single-GPU Epoch {epoch+1}: Vocabulary not ready yet")
                    continue

                # Create a simple test sequence for inference
                start_token = move_to_idx['<STARTGAME>']
                sample_tokens = [start_token] + [move_to_idx.get('41lpab', 0)] * 10  # Simple test sequence
                sample_x = torch.tensor([sample_tokens], dtype=torch.long).to(device)

                epoch_all_text = test_progress(
                    epoch, num_epochs, batch_idx, data_loader, avg_epoch_loss,
                    model, sample_x, 30, "", idx_to_move
                )
                print(f"🎯 Single-GPU Epoch {epoch+1} inference completed")
            except Exception as e:
                print(f"⚠️  Single-GPU Epoch {epoch+1} inference failed: {e}")
                import traceback
                traceback.print_exc()

            if len(epoch_losses) > 1:
                loss_change = epoch_losses[-2] - epoch_losses[-1]
                print(f"Loss change from previous epoch: {loss_change:+.4f}")

    # Final training summary
    if epoch_losses:
        initial_loss = epoch_losses[0]
        final_loss = epoch_losses[-1]
        total_improvement = initial_loss - final_loss
        print(f"\nTraining Summary:")
        print(f"Initial loss: {initial_loss:.4f}")
        print(f"Final loss: {final_loss:.4f}")
        print(f"Total improvement: {total_improvement:.4f}")
        print(f"Improvement rate: {total_improvement/len(epoch_losses):.4f} per epoch")

    print("Training completed!")


# Main training function
def train_backgammon_model():
    """Main training entry point"""
    # Single process mode - load data interactively
    text, checkpoint_data = load_data_interactive()
    _train_backgammon_model_core(text, checkpoint_data)


def predict_backgammon_moves(model, game_history, dice_roll, idx_to_move, move_to_idx, device, top_k=5):
    """
    Generate top-k move sequence predictions (pairs or quadruples).
    
    Returns:
        List of (joint_prob, move_sequence) tuples, sorted by probability
    """
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
        return pairs[:k]

    def get_top_k_quadruples(model, dice_tokens, idx_to_move, move_to_idx, device, k=10, game_history=[]):
        """
        Get top-k move quadruples with joint probabilities for doubles.
        Uses beam search to efficiently explore 4-move sequences.
        """
        model.eval()
        
        # Build full context
        input_tokens = game_history + dice_tokens
        context_indices = [move_to_idx[token] for token in input_tokens if token in move_to_idx]
        
        if len(context_indices) > model.block_size:
            context_indices = context_indices[-model.block_size:]
            
        # Beam search: keep top-k sequences at each step
        beam_width = k * 2  # Keep more candidates during search
        
        # (context, prob, moves_list)
        sequences = [(context_indices, 1.0, [])] 
        
        with torch.no_grad():
            # Doubles can have up to 4 moves
            for step in range(4): 
                candidates = []
                
                for context, prob, moves in sequences:
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
                    top_k_step = min(20, len(probs)) # Check top 20 next tokens
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
                                candidates.append((context + [idx.item()], joint_prob, ['<NOMOVE>']))
                            continue
                            
                        if token == '<EOM>':
                             # Valid partial sequence end
                             candidates.append((context + [idx.item()], joint_prob, moves))
                             continue
                             
                        if not token.startswith('m_'):
                            continue
                        
                        new_context = context + [idx.item()]
                        new_moves = moves + [token]
                        
                        candidates.append((new_context, joint_prob, new_moves))
                
                # Keep top beam_width sequences
                if not candidates:
                    break
                    
                candidates.sort(key=lambda x: x[1], reverse=True)
                
                next_sequences = []
                for ctx, p, m in candidates:
                     if len(m) > 0 and m[-1] == '<NOMOVE>':
                         # Finished
                         sequences.append((ctx, p, m)) 
                         pass
                     elif len(m) == step + 1: # Successfully added a move
                         next_sequences.append((ctx, p, m))
                
                sequences = next_sequences[:beam_width]
                
                if not sequences:
                    break

        sequences.sort(key=lambda x: x[1], reverse=True)
        
        # Convert to expected return format
        results = [(prob, moves) for _, prob, moves in sequences]
        return results[:k]

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


def predict_backgammon_move(model, game_history, dice_roll, idx_to_move, move_to_idx, device):
    """
    Predict the best backgammon move (legacy function, now calls predict_backgammon_moves).
    """
    results = predict_backgammon_moves(model, game_history, dice_roll, idx_to_move, move_to_idx, device, top_k=1)
    return results[0][0] if results else "No move predicted"


class BackgammonMovePredictor:
    """
    Production-ready interface for backgammon LLM move prediction.

    This class provides a clean API for game engines to:
    - Load trained models
    - Get strategic move recommendations
    - Handle illegal moves with automatic filtering
    - Integrate seamlessly with existing backgammon implementations

    Key Features:
    - Automatic model loading and device management
    - Full game context support (complete move history)
    - Top-k predictions with confidence scores
    - Legal move validation and filtering
    - Graceful fallback handling

    Usage:
        predictor = BackgammonMovePredictor(model_path="model.pth")
        moves = predictor.predict_moves(game_history, dice="52", top_k=5)
        legal_moves = predictor.get_legal_moves(game_history, dice="52",
                                               board_validator=is_legal_fn)
    """

    def __init__(self, model_path=None, device=None):
        """
        Initialize the move predictor.

        Args:
            model_path: Path to trained model checkpoint (if None, loads interactively)
            device: Torch device (auto-detected if None)
        """
        if device is None:
            # Use same device detection logic as main training code
            if platform.system() == "Darwin" and torch.backends.mps.is_available():
                device = torch.device('mps')
            elif torch.cuda.is_available():
                device = torch.device('cuda')
            else:
                device = torch.device('cpu')

        self.device = device
        self.model = None
        self.idx_to_move = None
        self.move_to_idx = None

        if model_path:
            self.load_model(model_path)
        else:
            # Interactive loading for development
            print("Loading model interactively...")
            loaded_data = load_model_file()
            if loaded_data:
                self.model, _, _, _, _, _, _, _, _, _, _, _, _, _ = loaded_data
                # Assume global dictionaries are set by load_model_file
                global move_to_idx, idx_to_move
                self.move_to_idx = move_to_idx
                self.idx_to_move = idx_to_move
                print("Model loaded successfully!")
            else:
                raise ValueError("Failed to load model")

    def load_model(self, model_path):
        """Load model from checkpoint file."""
        loaded_data = load_model_file(model_path)
        if loaded_data:
            self.model, _, _, _, _, _, _, _, _, _, _, _, _, _ = loaded_data
            # Assume global dictionaries are set
            global move_to_idx, idx_to_move
            self.move_to_idx = move_to_idx
            self.idx_to_move = idx_to_move
        else:
            raise ValueError(f"Failed to load model from {model_path}")

    def predict_moves(self, game_history, dice, top_k=5):
        """
        Get strategic move recommendations from the LLM.

        This method provides the core AI functionality:
        - Takes complete game context (all previous moves + dice)
        - Returns top-k move predictions ranked by model confidence
        - Filters out non-move predictions (dice, special tokens)
        - Enables fallback logic in game engines

        Args:
            game_history: Complete token sequence from game start
                         Example: ["<STARTGAME>", "d31", "m_0to3", "d52", "m_3to8"]
            dice: Current dice roll as string ("31" for 3,1)
            top_k: Number of top predictions to return (default 5 for fallback)

        Returns:
            List of (move_token, confidence) tuples sorted by confidence
            Example: [("m_12to15", 0.85), ("m_8to12", 0.12), ...]
        """
        if not self.model:
            raise ValueError("Model not loaded")

        # Get all predictions, then filter to only moves
        all_predictions = predict_backgammon_moves(
            self.model, game_history, dice,
            self.idx_to_move, self.move_to_idx,
            self.device, top_k=top_k*2  # Get more candidates to ensure we get enough moves
        )

        # Filter to only move tokens (m_*)
        move_predictions = [(token, confidence) for token, confidence in all_predictions
                          if token.startswith('m_')][:top_k]

        return move_predictions

    def get_legal_moves(self, game_history, dice, board_validator=None, top_k=5):
        """
        Get legal moves by filtering LLM predictions through board validation.

        This method combines AI prediction with game rule enforcement:
        1. Gets raw predictions from LLM (may include illegal moves)
        2. Filters through board_validator function to check legality
        3. Returns only legal moves, maintaining confidence ranking
        4. Ensures game engine compliance with backgammon rules

        Args:
            game_history: Complete game token sequence
            dice: Current dice roll string
            board_validator: Function(move_token) -> bool
                           Should return True if move is legal on current board
            top_k: Maximum legal moves to return

        Returns:
            List of legal (move_token, confidence) tuples, sorted by confidence
            Guaranteed to only contain moves that pass board validation

        Example:
            def is_legal(move_token):
                return check_move_against_current_board(move_token)

            legal = predictor.get_legal_moves(history, "52", is_legal, top_k=3)
        """
        all_predictions = self.predict_moves(game_history, dice, top_k=top_k*2)  # Get more candidates

        if board_validator is None:
            return all_predictions[:top_k]

        # Filter for legal moves
        legal_moves = []
        for move_token, confidence in all_predictions:
            if board_validator(move_token):
                legal_moves.append((move_token, confidence))
                if len(legal_moves) >= top_k:
                    break

        return legal_moves


def load_data_interactive():
    """Load data in single process mode (interactive)"""
    global gpu_indices
    text = ""
    checkpoint_data = None

    # Ask user to choose between loading model or creating new
    load_or_create = get_input_with_default("Load a model file or Create a new model? (l/c)", "c").lower()

    if load_or_create == 'l':
        loaded_data = load_model_file()
        if loaded_data:
            checkpoint_data = loaded_data
        else:
            print("Failed to load model. Creating a new one.")
            checkpoint_data = None
    else:
        checkpoint_data = None

    # GPU selection - only for fresh training
    if checkpoint_data is None and torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        all_gpus = list(range(num_gpus))

        print("\nGPU Selection:")
        print(f"System has {num_gpus} GPU{'s' if num_gpus > 1 else ''} available")
        print("Note: Enter your GPU number directly")
        custom_gpus = input(f"Enter GPU indices separated by commas (default: all {num_gpus} GPUs): ")

        if not custom_gpus.strip():
            gpu_indices = all_gpus
            print(f"Using all {num_gpus} GPU{'s' if num_gpus > 1 else ''}: {gpu_indices}")
        else:
            try:
                gpu_indices = [int(idx.strip()) for idx in custom_gpus.split(',')]
                print(f"Selected GPUs: {gpu_indices}")
            except ValueError:
                gpu_indices = all_gpus
                print(f"Invalid input. Using all {num_gpus} GPU{'s' if num_gpus > 1 else ''}: {gpu_indices}")

        # Set CUDA_VISIBLE_DEVICES
        if gpu_indices:
            os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, gpu_indices))
            print(f"CUDA_VISIBLE_DEVICES set to: {os.environ['CUDA_VISIBLE_DEVICES']}")

    # Load backgammon file
    file_path = create_file_dialog(title="Select Backgammon Games File for Training", filetypes=[("Text files", "*.txt")])
    if not file_path:
        print("No backgammon file selected. Exiting.")
        exit()

    print(f"Loading backgammon file: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    games = text.split('\n\n')
    games = [game.strip() for game in games if game.strip()]
    text = '\n'.join(games)  # Concatenate for efficient GPU training (like ChessBrain)
    print(f"Backgammon dataset loaded. Total games: {len(games)}, Total characters: {len(text)}")

    return text, checkpoint_data


def demo_move_predictor():
    """Demo function showing how to use the BackgammonMovePredictor."""
    print("🎮 Backgammon LLM Move Predictor Demo")
    print("=" * 50)

    # This would normally load a trained model
    print("This demo shows the interface - load a trained model to use it.")
    print()
    print("Example usage:")
    print("""
from BackgammonBrain_Plateau_11_1_25 import BackgammonMovePredictor

# Initialize with trained model
predictor = BackgammonMovePredictor(model_path="path/to/trained/model.pth")

# Game state after some moves
game_history = ["<STARTGAME>", "d31", "m_adln", "d52", "m_mhxw"]

# Player rolls 4,1
dice_roll = "41"

# Get top 3 move suggestions
moves = predictor.predict_moves(game_history, dice_roll, top_k=3)
print("Top 3 suggested moves:")
for move, confidence in moves:
    print(f"  {move}: {confidence:.1%} confidence")

# With board validation (would filter illegal moves)
def is_legal_move(move_token):
    # This would check against actual board position
    return True  # Placeholder

legal_moves = predictor.get_legal_moves(game_history, dice_roll,
                                       board_validator=is_legal_move, top_k=3)
print("\\nLegal moves only:")
for move, confidence in legal_moves:
    print(f"  {move}: {confidence:.1%} confidence")
    """)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "demo":
        demo_move_predictor()
    else:
        print("BackgammonBrain - Backgammon Move Prediction LLM")
        print("=" * 50)

        # Start training
        print("🖥️  Single process mode - interactive GUI setup")
        train_backgammon_model()
