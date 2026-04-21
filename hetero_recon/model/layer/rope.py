from timm.layers import (
    SwiGLUPacked as SwiGLU,
    to_2tuple,
)
import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import (
    Attention,
    Block,
)

__all__ = [
    "RoPEAttention",
    "RoPEBlock",
    "precompute_freqs_cis",
    "apply_rotary_emb",
]


def precompute_freqs_cis(
    grid_size: tuple[int, int],
    dim: int,
    theta: float = 10000.0,
    grid_offset: int | tuple[int, int] = 0,
) -> tuple[torch.Tensor]:
    """
    Precompute the trigonometric form of the base frequencies for the rotary positional encoding.

    Args:
        `grid_size` (tuple[int, int]): The grid size (H, W) of the input image.
        `dim` (int): The embedding dimension.
        `theta` (float): The base frequency.
        `grid_offset` (int): The offset of the grid.

    Returns:
        `freqs_cis` (torch.Tensor): The trigonometric form of the base frequencies.
    """
    grid_offset = to_2tuple(grid_offset)
    assert grid_offset[0] >= 0 and grid_offset[1] >= 0, "`grid_offset` should be non-negative."

    assert dim % 4 == 0, "`dim` should be divisible by 4."
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 4)[:(dim // 4)].float() / dim))                   # [E // 4]
    y, x = torch.meshgrid(
        torch.arange(grid_size[0], device=freqs.device, dtype=torch.float32),
        torch.arange(grid_size[1], device=freqs.device, dtype=torch.float32),
        indexing="ij",
    )                                                                                               # [H, W]
    freqs_y = torch.einsum("hw, e->hwe", y + grid_offset[0], freqs)                                 # [H, W, E // 4]
    freqs_x = torch.einsum("hw, e->hwe", x + grid_offset[1], freqs)                                 # [H, W, E // 4]
    return torch.cat([freqs_y.cos(), freqs_y.sin(), freqs_x.cos(), freqs_x.sin()], -1).unsqueeze(0) # [1, H, W, E]


def apply_rotary_emb(
    xq: torch.Tensor,        # [B, N, L, E]
    xk: torch.Tensor,        # [B, N, L, E]
    freqs_cis: torch.Tensor, # [B, L, E],
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Apply the rotary positional embedding to the query and key tensors.
    The RoPE we used here is the PaLM style RoPE.
    """

    xq_real_y, xq_imag_y, xq_real_x, xq_imag_x = xq.float().chunk(4, dim=-1) # [B, N, L, E // 4]
    xk_real_y, xk_imag_y, xk_real_x, xk_imag_x = xk.float().chunk(4, dim=-1) # [B, N, L, E // 4]
    cos_y, sin_y, cos_x, sin_x = freqs_cis.unsqueeze(1).chunk(4, dim=-1)     # [B, 1, L, E // 4]

    xq_out = torch.cat(
        [
            xq_real_y * cos_y - xq_imag_y * sin_y,
            xq_real_y * sin_y + xq_imag_y * cos_y,
            xq_real_x * cos_x - xq_imag_x * sin_x,
            xq_real_x * sin_x + xq_imag_x * cos_x,
        ],
        dim=-1,
    ).type_as(xq)                                  # [B, N, L, E]
    xk_out = torch.cat(
        [
            xk_real_y * cos_y - xk_imag_y * sin_y,
            xk_real_y * sin_y + xk_imag_y * cos_y,
            xk_real_x * cos_x - xk_imag_x * sin_x,
            xk_real_x * sin_x + xk_imag_x * cos_x,
        ],
        dim=-1,
    ).type_as(xk)                                  # [B, N, L, E]

    return xq_out, xk_out


class RoPEAttention(Attention):
    """
    Attention block with support for RoPE.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: nn.Module = nn.RMSNorm,
    ) -> None:
        super().__init__(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
        )

    def _prepare_qkv(self, x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        x_shape = x.shape
        B, L = x_shape[: 2]

        qkv = self.qkv(x).reshape(B, L, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)
        q, k = apply_rotary_emb(q, k, freqs_cis)

        return q, k, v, x_shape

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        q, k, v, x_shape = self._prepare_qkv(x, freqs_cis) # [B, N, L, E]

        if self.fused_attn:
            x = F.scaled_dot_product_attention(q, k, v, attn_mask, dropout_p=self.attn_drop.p if self.training else 0.0)
        else:
            attn = self._get_attn_map_qk(q, k, attn_mask)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(x_shape)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def get_attn_map(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        q, k, _, _ = self._prepare_qkv(x, freqs_cis)
        return self._get_attn_map_qk(q, k, attn_mask)


class RoPEBlock(Block):
    """
    Vision Transformer block with support for RoPE Attention and SwiGLU.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        proj_drop: float = 0.0,
        attn_drop: float = 0.0,
        init_values: float | None = None,
        drop_path: float = 0.0,
        act_layer: nn.Module = nn.SiLU,
        norm_layer: nn.Module = nn.RMSNorm,
        attn_layer: nn.Module = RoPEAttention,
        mlp_layer: nn.Module = SwiGLU,
    ) -> None:
        super().__init__(
            dim=dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            proj_drop=proj_drop,
            attn_drop=attn_drop,
            init_values=init_values,
            drop_path=drop_path,
            act_layer=act_layer,
            norm_layer=norm_layer,
            attn_layer=attn_layer,
            mlp_layer=mlp_layer,
        )

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), freqs_cis, attn_mask)))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x

    def get_attn_map(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.attn.get_attn_map(self.norm1(x), freqs_cis, attn_mask)
