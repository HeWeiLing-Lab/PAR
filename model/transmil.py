# src/models/transmil.py
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from nystrom_attention import NystromAttention
from src.utils import registry

class TransLayer(nn.Module):
    def __init__(self, dim: int = 512, heads: int = 8, dropout: float = 0.1, pinv_iterations: int = 6):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attn = NystromAttention(
            dim=dim,
            dim_head=dim // heads,
            heads=heads,
            num_landmarks=dim // 2,
            pinv_iterations=pinv_iterations,
            residual=True,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 1+L, C]
        return x + self.attn(self.norm(x))

class PPEG(nn.Module):
    def __init__(self, dim: int = 512):
        super().__init__()
        self.proj  = nn.Conv2d(dim, dim, 7, 1, 7 // 2, groups=dim)
        self.proj1 = nn.Conv2d(dim, dim, 5, 1, 5 // 2, groups=dim)
        self.proj2 = nn.Conv2d(dim, dim, 3, 1, 3 // 2, groups=dim)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        B, L, C = x.shape
        cls_token, feat = x[:, :1], x[:, 1:]                   # [B,1,C], [B,N,C]
        feat_2d = feat.transpose(1, 2).reshape(B, C, H, W)     # [B,C,H,W]
        y = self.proj(feat_2d) + feat_2d + self.proj1(feat_2d) + self.proj2(feat_2d)
        y = y.flatten(2).transpose(1, 2)                       # [B,N,C]
        return torch.cat([cls_token, y], dim=1)                # [B,1+N,C]

@registry.register("model", "TransMIL")
class TransMIL(nn.Module):
    def __init__(
        self,
        in_dim: int = 1536,
        num_classes: int = 2,
        dim: int = 512,
        heads: int = 8,
        dropout: float = 0.1,
        pinv_iterations: int = 6,
    ):
        super().__init__()
        self.n_classes = num_classes
        self.dim = dim

        self.fc_in = nn.Sequential(nn.Linear(in_dim, dim), nn.ReLU(inplace=True))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))

        self.layer1 = TransLayer(dim=dim, heads=heads, dropout=dropout, pinv_iterations=pinv_iterations)
        self.layer2 = TransLayer(dim=dim, heads=heads, dropout=dropout, pinv_iterations=pinv_iterations)

        self.pos_layer = PPEG(dim=dim)
        self.norm = nn.LayerNorm(dim)
        self.fc_out = nn.Linear(dim, num_classes if num_classes > 1 else 1)

    # ---------- helpers ----------
    def _pad_to_square_seq(self, h: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        B, N, C = h.shape
        side = int(math.ceil(N ** 0.5))
        N_pad = side * side
        add = N_pad - N
        if add > 0:
            h = torch.cat([h, h[:, :add, :]], dim=1)
        return h, side, side

    @staticmethod
    def _grid_from_coords(coords: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        xu, invx = torch.unique(coords[:, 0], sorted=True, return_inverse=True)
        yu, invy = torch.unique(coords[:, 1], sorted=True, return_inverse=True)
        Sx, Sy = int(xu.numel()), int(yu.numel())
        grid = coords.new_full((Sx, Sy), -1, dtype=torch.long)
        grid[invx, invy] = torch.arange(coords.size(0), device=coords.device, dtype=torch.long)
        return grid.reshape(-1), Sx, Sy

    def _arrange_with_coords(self, h: torch.Tensor, coords: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        B, N, C = h.shape
        flat_idx, H, W = self._grid_from_coords(coords)         
        valid = flat_idx >= 0
        h_pad = h.new_zeros((B, H * W, C))
        if valid.any():
            src = flat_idx[valid]                               
            h_pad[:, valid] = h[:, src, :]                      
        
        missing = (~valid).nonzero(as_tuple=False).squeeze(1)   
        if missing.numel() > 0 and N > 0:
           
            rep = (missing.numel() + N - 1) // N
            fill_idx = torch.arange(N, device=h.device).repeat(rep)[:missing.numel()]
            h_pad[:, missing] = h[:, fill_idx, :]
        return h_pad, H, W

    # ---------- forward ----------
    def forward(self, x: torch.Tensor, coords: torch.Tensor | None = None):
       
        single = False
        if x.dim() == 2:
            x = x.unsqueeze(0)  # [1,N,D]
            single = True
        B, N, _ = x.shape

        
        if coords is not None:
            if coords.dim() == 3 and coords.size(0) == 1:
                coords = coords.squeeze(0)
            coords = coords[:, :2].long().to(x.device)

        
        h = self.fc_in(x.float())                                # [B,N,dim]

        
        if coords is not None:
            h, H, W = self._arrange_with_coords(h, coords)       # [B,H*W,dim]
        else:
            h, H, W = self._pad_to_square_seq(h)                 # [B,H*W,dim]

  
        cls = self.cls_token.expand(B, 1, self.dim).to(h.device) # [B,1,dim]
        h = torch.cat([cls, h], dim=1)                           # [B,1+H*W,dim]

       
        h = self.layer1(h)
        h = self.pos_layer(h, H, W)
        h = self.layer2(h)

        h_cls = self.norm(h)[:, 0]                               # [B,dim]
        logits = self.fc_out(h_cls)                              

        return logits
