# src/models/abmil.py
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Union, Optional

try:
    from src.utils.registry import register
except Exception:
    def register(*args, **kwargs):
        def deco(cls): return cls
        return deco


class _ABMILCore(nn.Module):
    def __init__(
        self,
        input_dim: int,
        M: int = 500,
        L: int = 128,
        attention_branches: int = 1,   # K
        num_classes: int = 2,          # C
        gated: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.M = M
        self.L = L
        self.K = attention_branches
        self.C = num_classes
        self.gated = gated

        
        mlp: List[nn.Module] = [nn.Linear(input_dim, M), nn.ReLU(inplace=True)]
        if dropout and dropout > 0:
            mlp.append(nn.Dropout(p=dropout))
        self.instance_embed = nn.Sequential(*mlp)  

       
        if gated:
            self.attention_V = nn.Sequential(nn.Linear(M, L), nn.Tanh())
            self.attention_U = nn.Sequential(nn.Linear(M, L), nn.Sigmoid())
            self.attention_w = nn.Linear(L, self.K)  
        else:
            self.attention = nn.Sequential(
                nn.Linear(M, L),
                nn.Tanh(),
                nn.Linear(L, self.K),               
            )

        
        self.classifier = nn.Linear(M * self.K, self.C)

    @torch.no_grad()
    def _ensure_2d(self, x: torch.Tensor) -> torch.Tensor:
       
        if x.dim() == 3:
            assert x.size(0) == 1, "only single bag。"
            x = x.squeeze(0)
        if x.dim() != 2:
            raise ValueError(f"期望 [N,D] 或 [1,N,D]，实际 {tuple(x.shape)}")
        return x

    def _forward_single_bag(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self._ensure_2d(x)            # [N, D]
        H = self.instance_embed(x)        # [N, M]

        if self.gated:
            A_v = self.attention_V(H)     
            A_u = self.attention_U(H)     
            A = self.attention_w(A_v * A_u)  
        else:
            A = self.attention(H)         

        A = A.transpose(1, 0)            
        A = F.softmax(A, dim=1)           

        Z = torch.mm(A, H)                
        Z = Z.reshape(1, self.K * self.M) 
        logits = self.classifier(Z)       
        return logits, A                  

    def forward(self, x: Union[torch.Tensor, List[torch.Tensor]]):
        
        if isinstance(x, list):
            logits_list, att_list = [], []
            for xi in x:
                li, Ai = self._forward_single_bag(xi)
                logits_list.append(li)
                att_list.append(Ai)
            logits = torch.cat(logits_list, dim=0)  # [B, C]
            return logits, att_list

        if x.dim() == 3 and x.size(0) > 1:   # [B, N, D]
            logits_list, att_list = [], []
            for i in range(x.size(0)):
                li, Ai = self._forward_single_bag(x[i])
                logits_list.append(li)
                att_list.append(Ai)
            logits = torch.cat(logits_list, dim=0)  # [B, C]
            return logits, att_list

        logits, A = self._forward_single_bag(x)
        return logits, [A]


@register("model", "ABMIL")
class ABMIL(_ABMILCore):
    
    def __init__(
        self,
        input_dim: int,
        M: int = 500,
        L: int = 128,
        attention_branches: int = 1,
        num_classes: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__(
            input_dim=input_dim,
            M=M,
            L=L,
            attention_branches=attention_branches,
            num_classes=num_classes,
            gated=False,
            dropout=dropout,
        )


@register("model", "GatedABMIL")
class GatedABMIL(_ABMILCore):
    
    def __init__(
        self,
        input_dim: int,
        M: int = 500,
        L: int = 128,
        attention_branches: int = 1,
        num_classes: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__(
            input_dim=input_dim,
            M=M,
            L=L,
            attention_branches=attention_branches,
            num_classes=num_classes,
            gated=True,
            dropout=dropout,
        )
