import torch
import torch.nn as nn
from typing import Optional, Dict, Any
from .layers import RMSNorm, PaniniRoPE
from .block import TransformerBlock

class PaniniConfig:
    """
    Configuration class to store custom model hyperparameters.
    Mathematical specifications are based on advanced transformer standard layouts.
    """
    def __init__(
        self,
        vocab_size: int = 64000,       # Matched to your custom Panini Tokenizer max capacity
        dim: int = 4096,               # Hidden structural dimension size
        n_layers: int = 32,            # Number of repeatable transformer blocks
        n_heads: int = 32,             # Number of query attention heads
        n_kv_heads: int = 8,           # Number of key/value heads for GQA efficiency
        max_seq_len: int = 8192,       # Maximum token context window limit
        norm_eps: float = 1e-5,        # Small stability factor for RMSNorm calculations
    ):
        self.vocab_size = vocab_size
        self.dim = dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = dim // n_heads
        self.max_seq_len = max_seq_len
        self.norm_eps = norm_eps


class PaniniTransformer(nn.Module):
    """
    The full Panini Core Transformer model network.
    Orchestrates dense word embeddings, stacked blocks, and final predictions.
    """
    def __init__(self, config: PaniniConfig):
        super().__init__()
        self.config = config
        
        # 1. Word Token Embedding Layer
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.dim)
        
        # 2. Shared Rotary Position Embeddings (RoPE with base frequency theta=500000.0)
        self.rope = PaniniRoPE(
            dim=config.head_dim, 
            max_seq_len=config.max_seq_len, 
            theta=500000.0
        )
        
        # 3. Stacked Custom Processing Blocks
        self.layers = nn.ModuleList([
            TransformerBlock(
                dim=config.dim,
                n_heads=config.n_heads,
                n_kv_heads=config.n_kv_heads,
                head_dim=config.head_dim
            )
            for _ in range(config.n_layers)
        ])
        
        # 4. Final Output Stabilization Norm and Prediction Head
        self.norm = RMSNorm(dim=config.dim, eps=config.norm_eps)
        self.output = nn.Linear(config.dim, config.vocab_size, bias=False)

    def _build_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        Creates an upper triangular masking screen to prevent the network 
        from reading ahead during learning loops.
        """
        mask = torch.full((seq_len, seq_len), float("-inf"), device=device)
        mask = torch.triu(mask, diagonal=1)
        return mask

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        # tokens shape layout: (batch_size, seq_len)
        _ , seq_len = tokens.shape
        
        # Step 1: Translate token identity numbers into dense vectors
        h = self.tok_embeddings(tokens)
        
        # Step 2: Build causal matrix masking boundary walls
        mask = self._build_causal_mask(seq_len, tokens.device)
        
        # Step 3: Stream vectors sequentially through your stacked network blocks
        for layer in self.layers:
            h = layer(h, self.rope, mask)
            
        # Step 4: Final stabilization layer normalization and vocabulary projection
        h = self.norm(h)
        logits = self.output(h) # final shape layout: (batch_size, seq_len, vocab_size)
        
        return logits
