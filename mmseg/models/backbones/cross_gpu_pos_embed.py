"""
Cross-GPU Position Embedding (CPE) for Spatial Parallel Attention.

Problem: In SP attention, after all_to_all, each GPU sees tokens from ALL GPUs,
but the attention has NO idea where each token is located in the full image.
For example, GPU0 (left half) and GPU1 (right half) tokens get mixed, but there's
no signal to distinguish "this token is from the left boundary" vs "right boundary".

Solution: Before spattn, inject absolute position embeddings based on each token's
GLOBAL position in the full image. This lets the cross-GPU attention reason about
spatial relationships (e.g., water body continuity across the GPU boundary).
"""

import math
import torch
import torch.nn as nn
import torch.distributed as dist


def _get_global_offsets(H_local, W_local, device):
    """Compute global h_start, w_start for this GPU based on rank."""
    if dist.is_initialized():
        world_size = dist.get_world_size()
        rank = dist.get_rank()
    else:
        world_size = 1
        rank = 0
    sp_times = int(math.log2(max(world_size, 1)))
    sp_h = 2 ** (sp_times // 2)
    sp_w = 2 ** (sp_times - sp_times // 2)
    rank_h = rank % sp_h
    rank_w = rank // sp_h
    h_start = rank_h * H_local
    w_start = rank_w * W_local
    h_idx = torch.arange(h_start, h_start + H_local, device=device)
    w_idx = torch.arange(w_start, w_start + W_local, device=device)
    return h_idx, w_idx


class CrossGPUPosEmbedSinusoidal(nn.Module):
    """Fixed sinusoidal cross-GPU position embedding with learnable scale.

    Uses standard sine/cosine encoding based on global token positions,
    but adds a learnable scale parameter to control the strength of position signal.
    """

    def __init__(self, embed_dim: int, max_h: int = 64, max_w: int = 64):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_h = max_h
        self.max_w = max_w

    def _sinusoidal_1d(self, indices: torch.Tensor, dim: int) -> torch.Tensor:
        """Generate sinusoidal encoding for a sequence of indices. [N, dim]"""
        half = dim // 2
        freq = torch.arange(half, device=indices.device, dtype=torch.float32)
        freq = 1.0 / (10000 ** (freq / half))                # [half]
        angles = indices.float().unsqueeze(1) * freq.unsqueeze(0)  # [N, half]
        return torch.cat([angles.sin(), angles.cos()], dim=-1)     # [N, dim]

    def forward(self, hw_shape: tuple, device: torch.device) -> torch.Tensor:
        H_local, W_local = hw_shape
        h_idx, w_idx = _get_global_offsets(H_local, W_local, device)

        h_emb = self._sinusoidal_1d(h_idx, self.embed_dim // 2)  # [H, C//2]
        w_emb = self._sinusoidal_1d(w_idx, self.embed_dim // 2)  # [W, C//2]

        h_emb = h_emb.unsqueeze(1).expand(-1, W_local, -1)  # [H, W, C//2]
        w_emb = w_emb.unsqueeze(0).expand(H_local, -1, -1)  # [H, W, C//2]
        pos = torch.cat([h_emb, w_emb], dim=-1)              # [H, W, C]
        return pos.view(1, H_local * W_local, -1)             # [1, N, C]


class CrossGPUPosEmbed(nn.Module):
    """
    Learnable absolute position embedding for cross-GPU spatial attention.

    Each GPU's tokens get position embeddings based on their location in the
    FULL (multi-GPU) image, not just the local half. This way, after all_to_all
    communication, the attention knows the global spatial position of every token.

    Args:
        embed_dim: Feature dimension (must match backbone stage output dim)
        max_h: Maximum feature map height (for full image at this stage)
        max_w: Maximum feature map width (for full image at this stage)
    """

    def __init__(self, embed_dim: int, max_h: int = 64, max_w: int = 64):
        super().__init__()
        # Separate learnable embeddings for height and width
        self.h_embed = nn.Embedding(max_h, embed_dim // 2)
        self.w_embed = nn.Embedding(max_w, embed_dim // 2)
        # Project concatenated h+w embeddings to full embed_dim
        self.proj = nn.Linear(embed_dim, embed_dim, bias=False)

        nn.init.trunc_normal_(self.h_embed.weight, std=0.02)
        nn.init.trunc_normal_(self.w_embed.weight, std=0.02)
        nn.init.eye_(self.proj.weight)  # identity init: don't distort features initially

        self.max_h = max_h
        self.max_w = max_w

    def forward(self, hw_shape: tuple, device: torch.device) -> torch.Tensor:
        """
        Generate position embeddings for this GPU's local tokens.

        The position indices are computed based on:
        - hw_shape: local feature map size (H_local, W_local)
        - dist.get_rank(): which GPU this is (determines global offset)
        - dist.get_world_size(): total GPUs (determines sp_h, sp_w)

        Returns:
            pos: [1, H_local * W_local, embed_dim] position embedding
        """
        H_local, W_local = hw_shape
        h_idx, w_idx = _get_global_offsets(H_local, W_local, device)
        h_idx = h_idx.clamp(0, self.max_h - 1)
        w_idx = w_idx.clamp(0, self.max_w - 1)

        # Look up embeddings
        h_emb = self.h_embed(h_idx)  # [H_local, embed_dim//2]
        w_emb = self.w_embed(w_idx)  # [W_local, embed_dim//2]

        # Broadcast and combine: [H_local, W_local, embed_dim]
        h_emb = h_emb.unsqueeze(1).expand(-1, W_local, -1)  # [H, W, C//2]
        w_emb = w_emb.unsqueeze(0).expand(H_local, -1, -1)  # [H, W, C//2]
        pos = torch.cat([h_emb, w_emb], dim=-1)             # [H, W, C]

        # Flatten to sequence and project
        pos = pos.view(1, H_local * W_local, -1)  # [1, N, C]
        pos = self.proj(pos)                        # [1, N, C]

        return pos
