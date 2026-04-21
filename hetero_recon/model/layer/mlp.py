from timm.layers import SwiGLUPacked as SwiGLU
import torch
import torch.nn as nn


class SwiGLUResidualLinear(nn.Module):

    def __init__(self, dim: int, mlp_ratio: float) -> None:
        super().__init__()

        self.norm = nn.LayerNorm(dim)
        self.mlp = SwiGLU(in_features=dim, hidden_features=int(dim * mlp_ratio), out_features=dim)
        self.init_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.mlp(self.norm(x))
        return x

    def init_weights(self):
        self.mlp.init_weights()


class SwiGLUResidualLinearMLP(nn.Module):

    def __init__(self, in_dim: int, depth: int, embed_dim: int, out_dim: int, mlp_ratio: float):
        super().__init__()
        self.mlp = nn.Sequential(
            *[
                nn.Linear(in_dim, embed_dim),
                *[SwiGLUResidualLinear(embed_dim, mlp_ratio) for _ in range(depth)],
                nn.Linear(embed_dim, out_dim),
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)
