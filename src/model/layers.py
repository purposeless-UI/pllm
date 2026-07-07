import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization (RMSNorm).
    Stabilizes training by scaling activations along the last dimension.
    """
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        # Calculate RMS along the last dimension
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class PaniniRoPE(nn.Module):
    """
    Rotary Position Embeddings (RoPE) optimized for the Panini architecture.
    Uses a base frequency of 500,000 for stable extended context windows.
    """
    def __init__(self, dim: int, max_seq_len: int = 8192, theta: float = 500000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.theta = theta
        
        # Compute inverse frequencies: theta^(-2i/d)
        inv_freq = 1.0 / (self.theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        
        # Precompute frequencies matrix
        self._set_cos_sin_cache(max_seq_len)

    def _set_cos_sin_cache(self, seq_len: int):
        t = torch.arange(seq_len, dtype=self.inv_freq.dtype)
        # Outer product to get frequency matrix: (seq_len, dim // 2)
        freqs = torch.outer(t, self.inv_freq)
        
        # Duplicate columns to match the standard RoPE implementation shape
        emb = torch.cat((freqs, freqs), dim=-1)
        
        self.register_buffer("cos_cached", emb.cos(), persistent=False) # (max_seq_len, dim)
        self.register_buffer("sin_cached", emb.sin(), persistent=False) # (max_seq_len, dim)

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        # Split features in half and swap with negative sign for the second half
        x1 = x[..., :self.dim // 2]
        x2 = x[..., self.dim // 2:]
        return torch.cat((-x2, x1), dim=-1)

    def forward(self, x: torch.Tensor, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Returns sliced caches matching current sequence length
        # Shape: (seq_len, dim)
        return self.cos_cached[:seq_len, :], self.sin_cached[:seq_len, :]

    def apply_rope(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, seq_len, num_heads, head_dim)
        # cos/sin shape: (seq_len, head_dim) -> Unsqueeze to broadcast across batch and heads
        cos = cos.unsqueeze(0).unsqueeze(2) # (1, seq_len, 1, head_dim)
        sin = sin.unsqueeze(0).unsqueeze(2) # (1, seq_len, 1, head_dim)
        
        # Standard RoPE mathematical formulation: R = X * cos(W) + rotate_half(X) * sin(W)
        return (x * cos) + (self._rotate_half(x) * sin)


class FeedForward(nn.Module):
    """
    SwiGLU Position-Wise Feed-Forward Network.
    Uses three linear projections to process non-linear tokens.
    """
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False) # Gate projection
        self.w2 = nn.Linear(hidden_dim, dim, bias=False) # Down projection
        self.w3 = nn.Linear(dim, hidden_dim, bias=False) # Up projection

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU activation formula: Swish(W1(x)) * W3(x)
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class GroupedQueryAttention(nn.Module):
    """
    Grouped-Query Attention (GQA) implementation.
    Reduces KV-cache memory bandwidth pressures across parallel processor systems.
    """
    def __init__(self, dim: int, n_heads: int, n_kv_heads: int, head_dim: int):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.num_queries_per_kv = n_heads // n_kv_heads
        
        self.q_proj = nn.Linear(dim, n_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * head_dim, dim, bias=False)

    def _repeat_kv(self, x: torch.Tensor, n_rep: int) -> torch.Tensor:
        # If 1 query per KV head, this is multi-query attention (MQA); no change needed
        if n_rep == 1:
            return x
        bs, seq_len, n_kv_heads, head_dim = x.shape
        # Expand and repeat KV heads along the head axis to match query heads count
        return (
            x[:, :, :, None, :]
            .expand(bs, seq_len, n_kv_heads, n_rep, head_dim)
            .reshape(bs, seq_len, n_kv_heads * n_rep, head_dim)
        )

    def forward(
        self, 
        x: torch.Tensor, 
        rope: PaniniRoPE, # UPDATED: Matches the custom class identity name
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        bs, seq_len, _ = x.shape
        
        # Project states to Q, K, V space
        q = self.q_proj(x).view(bs, seq_len, self.n_heads, self.head_dim)
        k = self.k_proj(x).view(bs, seq_len, self.n_kv_heads, self.head_dim)
        v = self.v_proj(x).view(bs, seq_len, self.n_kv_heads, self.head_dim)
        
        # Fetch and apply Rotary Position Embeddings
        cos, sin = rope(k, seq_len)
        q = rope.apply_rope(q, cos, sin)
        k = rope.apply_rope(k, cos, sin)
        
        # Expand Keys and Values if Grouped-Query Attention is utilized
        k = self._repeat_kv(k, self.num_queries_per_kv)
        v = self._repeat_kv(v, self.num_queries_per_kv)
        
        # Transpose matrices for batched matrix multiplication: (bs, n_heads, seq_len, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Scaled Dot-Product Attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))
        
        if mask is not None:
            scores = scores + mask
            
        scores = F.softmax(scores.float(), dim=-1).type_as(q)
        output = torch.matmul(scores, v) # (bs, n_heads, seq_len, head_dim)
        
        # Restore layout to original sequence dimension shape
        output = output.transpose(1, 2).contiguous().view(bs, seq_len, -1)
        
        return self.o_proj(output)
