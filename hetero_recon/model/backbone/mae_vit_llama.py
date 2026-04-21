from functools import partial
import math
from typing import Sequence

from timm.layers import (
    LayerType,
    PatchEmbed,
    SwiGLUPacked as SwiGLU,
)
import torch
import torch.nn as nn

from coach_pl.configuration import configurable

from .backbone import Backbone
from .build import BACKBONE_REGISTRY
from hetero_recon.model.layer import (
    precompute_freqs_cis,
    RoPEAttention as Attention,
    RoPEBlock as Block,
)

__all__ = ["MAEViTLlama"]


@BACKBONE_REGISTRY.register()
class MAEViTLlama(Backbone):
    """
    Llama style Vision Transformer backbone pretrained by MAE.
    Note that absolute `pos_embed` is replaced with `RoPE`.
    """

    @configurable
    def __init__(
        self,
        img_size: int | tuple[int, int] = 224,
        patch_size: int | tuple[int, int] = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_norm: bool = False,
        drop_path_rate: float = 0.0,
        final_norm: bool = True,
        dynamic_img_size: bool = False,
        dynamic_img_pad: bool = False,
        num_register_tokens: int = 4,
        embed_layer: LayerType = PatchEmbed,
        block_fn: LayerType = Block,
        act_layer: LayerType = nn.SiLU,
        norm_layer: LayerType = partial(nn.RMSNorm, eps=1e-6),
        attn_layer: LayerType = Attention,
        mlp_layer: LayerType = SwiGLU,
    ) -> None:
        super().__init__(
            in_channels=in_chans,
            out_channels=embed_dim,
            stride=patch_size,
        )

        assert num_register_tokens >= 0
        self.dynamic_img_size = dynamic_img_size
        self.num_prefix_tokens = num_register_tokens + 1
        self._head_dim = embed_dim // num_heads

        self.patch_embed = embed_layer(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            output_fmt="NHWC",
            strict_img_size=not dynamic_img_size,
            dynamic_img_pad=dynamic_img_pad,
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.reg_tokens = nn.Parameter(torch.zeros(1, num_register_tokens, embed_dim)) if num_register_tokens > 0 else None

        grid_size = self.patch_embed.grid_size
        self.register_buffer(
            "prefix_tokens_freqs_cis",
            precompute_freqs_cis((1, 1), self._head_dim).reshape(1, 1, -1).repeat(1, self.num_prefix_tokens, 1),
        )                                                                                                        # [1, num_prefix_tokens, head_dim]
        self.register_buffer(
            "freqs_cis",
            precompute_freqs_cis(grid_size, self._head_dim, grid_offset=1).reshape(1, math.prod(grid_size), -1),
        )                                                                                                        # [1, H * W, head_dim]

        # Stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        self.blocks = nn.ModuleList(
            block_fn(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_norm=qk_norm,
                drop_path=dpr[i],
                act_layer=act_layer,
                norm_layer=norm_layer,
                attn_layer=attn_layer,
                mlp_layer=mlp_layer,
            ) for i in range(depth)
        )
        self.norm = norm_layer(embed_dim) if final_norm else nn.Identity()

        self.init_weights()

    def init_weights(self) -> None:
        nn.init.normal_(self.cls_token, std=1e-6)
        if self.reg_tokens is not None:
            nn.init.normal_(self.reg_tokens, std=1e-6)

        self.apply(self._init_weights)

    def _prepare_tokens(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, int, int]:
        B, H, W, E = x.shape
        x = x.reshape(B, H * W, E) # [B, H * W, E]

        # Precompute freqs_cis
        if self.dynamic_img_size:
            freqs_cis = precompute_freqs_cis((H, W), self._head_dim, grid_offset=1).reshape(1, H * W, -1).to(x.device)
        else:
            freqs_cis = self.freqs_cis # [1, H * W, E']
        freqs_cis = freqs_cis.expand(B, -1, -1)

        # Concatenate cls_token and reg_tokens with patch tokens
        cls_token = self.cls_token.expand(B, 1, E)
        if self.reg_tokens is not None:
            reg_tokens = self.reg_tokens.expand(B, -1, E)
            x = torch.cat([cls_token, reg_tokens, x], dim=1) # [B, num_prefix_tokens + H * W, E]
        else:
            x = torch.cat([cls_token, x], dim=1)             # [B, num_prefix_tokens + H * W, E]

        # Concatenate prefix tokens with freqs_cis
        prefix_tokens_freqs_cis = self.prefix_tokens_freqs_cis.expand(B, -1, -1)
        freqs_cis = torch.cat([prefix_tokens_freqs_cis, freqs_cis], dim=1)

        return x, freqs_cis, H, W

    def _image_to_tokens(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, int, int]:
        x = self.patch_embed(x)                      # [B, H, W, E]
        x, freqs_cis, H, W = self._prepare_tokens(x) # [B, num_prefix_tokens + H * W, E]
        return x, freqs_cis, H, W

    def _tokens_to_features(self, x: torch.Tensor, freqs_cis: torch.Tensor, apply_norm: bool = True) -> torch.Tensor:
        for block in self.blocks:
            x = block(x, freqs_cis)
        if apply_norm:
            x = self.norm(x)
        return x

    def forward(self, x: torch.Tensor, apply_norm: bool = True) -> dict[str, torch.Tensor | int]:
        x, freqs_cis, H, W = self._image_to_tokens(x)
        x = self._tokens_to_features(x, freqs_cis, apply_norm)
        return {
            "x_norm_clstoken": x[:, 0],
            "x_norm_regtokens": x[:, 1 : self.num_prefix_tokens],
            "x_norm_patchtokens": x[:, self.num_prefix_tokens :],
            "patch_tokens_h": H,
            "patch_tokens_w": W,
        }

    def get_intermediate_features(
        self,
        x: torch.Tensor,
        block_taken_indices: Sequence[int],
        reshape: bool = True,
        return_class_token: bool = False,
        apply_norm: bool = True,
    ) -> Sequence[tuple[torch.Tensor, torch.Tensor | None]]:
        x, freqs_cis, H, W = self._image_to_tokens(x)

        # Extract intermediate features
        outputs = []
        for i, block in enumerate(self.blocks):
            x = block(x, freqs_cis)
            if i in block_taken_indices:
                outputs.append(x)
        assert len(outputs) == len(block_taken_indices), f"Expected {len(block_taken_indices)} outputs, but got {len(outputs)}"

        # Apply layer norm to all the intermediate features
        if apply_norm:
            outputs = [self.norm(out) for out in outputs]

        if return_class_token:
            class_tokens = [out[:, 0] for out in outputs]
        else:
            class_tokens = [None for _ in outputs]

        outputs = [out[:, self.num_prefix_tokens :] for out in outputs]
        if reshape:
            B = x.size(0)
            outputs = [out.reshape(B, H, W, -1) for out in outputs]

        return tuple(zip(outputs, class_tokens))

    def get_intermediate_attn_map(self, x: torch.Tensor, block_taken_indices: Sequence[int]) -> Sequence[torch.Tensor]:
        x, freqs_cis, _, _ = self._image_to_tokens(x)

        # Extract intermediate attention map
        output = []
        for i, block in enumerate(self.blocks):
            if i in block_taken_indices:
                output.append(block.get_attn_map(x, freqs_cis))
            x = block(x, freqs_cis)
        assert len(output) == len(block_taken_indices), f"Expected {len(block_taken_indices)} outputs, but got {len(output)}"

        return output


def draco_llama_small(
    img_size: int,
    patch_size: int,
    in_chans: int,
    final_norm: bool,
    dynamic_img_size: bool,
    dynamic_img_pad: bool,
) -> MAEViTLlama:
    return MAEViTLlama(
        img_size=img_size,
        patch_size=patch_size,
        in_chans=in_chans,
        final_norm=final_norm,
        dynamic_img_size=dynamic_img_size,
        dynamic_img_pad=dynamic_img_pad,
        **Backbone.get_scale("small"),
    )


def draco_llama_base(
    img_size: int,
    patch_size: int,
    in_chans: int,
    final_norm: bool,
    dynamic_img_size: bool,
    dynamic_img_pad: bool,
) -> MAEViTLlama:
    return MAEViTLlama(
        img_size=img_size,
        patch_size=patch_size,
        in_chans=in_chans,
        final_norm=final_norm,
        dynamic_img_size=dynamic_img_size,
        dynamic_img_pad=dynamic_img_pad,
        **Backbone.get_scale("base"),
    )
