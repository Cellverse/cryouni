from timm.layers import (
    DropPath,
    LayerScale,
    Mlp,
    SwiGLU,
    SwiGLUPacked,
)
from timm.models.vision_transformer import Attention as TimmAttention
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "Attention",
    "Block",
]


class Attention(TimmAttention):

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
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

    def _prepare_qkv(self, x: torch.Tensor) -> torch.Tensor:
        x_shape = x.shape
        B, L = x_shape[: 2]

        qkv = self.qkv(x).reshape(B, L, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        return q, k, v, x_shape

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        q, k, v, x_shape = self._prepare_qkv(x)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(q, k, v, attn_mask, dropout_p=self.attn_drop.p if self.training else 0.)
        else:
            attn = self._get_attn_map_qk(q, k, attn_mask)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(x_shape)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def get_attn_map(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        q, k, _, _ = self._prepare_qkv(x)
        return self._get_attn_map_qk(q, k, attn_mask)

    def _get_attn_map_qk(self, q: torch.Tensor, k: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        if attn_mask is not None:
            attn = attn.masked_fill(attn_mask, float("-inf"))
        attn = attn.softmax(dim=-1)
        return attn


class Block(nn.Module):

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
        act_layer: nn.Module = nn.GELU,
        norm_layer: nn.Module = nn.LayerNorm,
        attn_layer: nn.Module = Attention,
        mlp_layer: nn.Module = Mlp,
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = attn_layer(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
        )
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        # Dynamically adjust the hidden features of the MLP layer.
        if mlp_layer == SwiGLU:
            hidden_features = int(dim * mlp_ratio * 2 / 3)
        elif mlp_layer == SwiGLUPacked:
            hidden_features = int(dim * mlp_ratio * 4 / 3)
        else:
            hidden_features = int(dim * mlp_ratio)

        self.norm2 = norm_layer(dim)
        self.mlp = mlp_layer(
            in_features=dim,
            hidden_features=hidden_features,
            act_layer=act_layer,
            drop=proj_drop,
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), attn_mask)))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x

    def get_attn_map(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        return self.attn.get_attn_map(self.norm1(x), attn_mask)
