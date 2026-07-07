import torch
import torch.nn as nn
from typing import Optional
from .layers import RMSNorm, GroupedQueryAttention, FeedForward, PaniniRoPE

class TransformerBlock(nn.Module):
    """
    A single custom Panini Transformer Layer Block.
    Combines pre-layer Normalization, Grouped-Query Attention, and SwiGLU 
    Feed-Forward sub-networks using high-stability residual streams.
    """
    def __init__(self, dim: int, n_heads: int, n_kv_heads: int, head_dim: int, multiple_of: int = 256):
        super().__init__()
        self.n_heads = n_heads
        self.dim = dim
        
        # 1. Grouped-Query Attention (GQA) Block Components
        self.attention_norm = RMSNorm(dim=dim)
        self.attention = GroupedQueryAttention(
            dim=dim,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            head_dim=head_dim
        )
        
        # 2. SwiGLU Feed-Forward Network (FFN) Block Components
        # Sizing formula matches high-performance computing memory layout parameters
        hidden_dim = int(2 * (4 * dim) / 3)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        
        self.ffn_norm = RMSNorm(dim=dim)
        self.feed_forward = FeedForward(dim=dim, hidden_dim=hidden_dim)

    def forward(
        self, 
        x: torch.Tensor, 
        rope: PaniniRoPE, 
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Processes hidden vector streams through attention and reasoning networks.
        
        Execution Flow:
        1. h = x + GroupedQueryAttention(RMSNorm(x))
        2. out = h + SwiGLU_FeedForward(RMSNorm(h))
        """
        # --- Step 1: High-Speed Attention Processing Stream ---
        attention_output = self.attention(self.attention_norm(x), rope, mask)
        h = x + attention_output
        
        # --- Step 2: Non-Linear SwiGLU Feed-Forward Processing Stream ---
        ffn_output = self.feed_forward(self.ffn_norm(h))
        out = h + ffn_output
        
        return out
