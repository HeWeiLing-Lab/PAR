# src/models/abmil.py
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Union, Optional

try:
    # 可选：如果你有注册表，自动注册，便于 CLI 按名称构建
    from src.utils.registry import register
except Exception:
    def register(*args, **kwargs):
        def deco(cls): return cls
        return deco


class _ABMILCore(nn.Module):
    """
    向量版 ABMIL / Gated-ABMIL 核心：
    - 输入：bag 实例向量 x ∈ R^{N×D}
    - 实例嵌入：Linear(D→M)+ReLU(+Dropout)
    - 注意力：    vanilla:  softmax( w^T tanh( V H ) )
               gated:    softmax( w^T (tanh(VH) ⊙ sigm(UH)) )
    - 聚合：Z = A H ∈ R^{K×M} 展平 → 分类器 Linear(KM→C)
    - 输出：logits ∈ R^{B×C}（不做 Sigmoid/Softmax），attn 列表（每个 bag：K×N）
    """
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

        # 实例嵌入（替代原始 Conv2d 特征提取）
        mlp: List[nn.Module] = [nn.Linear(input_dim, M), nn.ReLU(inplace=True)]
        if dropout and dropout > 0:
            mlp.append(nn.Dropout(p=dropout))
        self.instance_embed = nn.Sequential(*mlp)  # [N, D] -> [N, M]

        # 注意力分支
        if gated:
            self.attention_V = nn.Sequential(nn.Linear(M, L), nn.Tanh())
            self.attention_U = nn.Sequential(nn.Linear(M, L), nn.Sigmoid())
            self.attention_w = nn.Linear(L, self.K)  # [N, L] -> [N, K]
        else:
            self.attention = nn.Sequential(
                nn.Linear(M, L),
                nn.Tanh(),
                nn.Linear(L, self.K),               # [N, L] -> [N, K]
            )

        # 分类头（输出 logits；训练时用 CE 或 BCEWithLogits）
        self.classifier = nn.Linear(M * self.K, self.C)

    @torch.no_grad()
    def _ensure_2d(self, x: torch.Tensor) -> torch.Tensor:
        # 允许 [1, N, D] / [N, D] -> [N, D]
        if x.dim() == 3:
            assert x.size(0) == 1, "传入 [B,N,D] 时请在外层按 bag 逐个处理；这里只处理单 bag。"
            x = x.squeeze(0)
        if x.dim() != 2:
            raise ValueError(f"期望 [N,D] 或 [1,N,D]，实际 {tuple(x.shape)}")
        return x

    def _forward_single_bag(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        单个 bag 前向：
        输入 x: [N, D]
        返回 (logits[1, C], A[K, N])
        """
        x = self._ensure_2d(x)            # [N, D]
        H = self.instance_embed(x)        # [N, M]

        if self.gated:
            A_v = self.attention_V(H)     # [N, L]
            A_u = self.attention_U(H)     # [N, L]
            A = self.attention_w(A_v * A_u)  # [N, K]
        else:
            A = self.attention(H)         # [N, K]

        A = A.transpose(1, 0)             # [K, N]
        A = F.softmax(A, dim=1)           # K heads，沿 N 归一化

        Z = torch.mm(A, H)                # [K, M]
        Z = Z.reshape(1, self.K * self.M) # [1, K*M]
        logits = self.classifier(Z)       # [1, C]
        return logits, A                  # logits 未过 Sigmoid/Softmax

    def forward(self, x: Union[torch.Tensor, List[torch.Tensor]]):
        """
        支持：
          - x: [N, D]
          - x: [1, N, D]
          - x: [B, N, D]  （将按 bag 逐个处理，支持异长）
          - x: list[Tensor[N_i, D]] （最通用异长）
        返回：
          logits: [B, C]
          atts:   List[Tensor[K, N_i]]
        """
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

        # [N, D] 或 [1, N, D]
        logits, A = self._forward_single_bag(x)
        return logits, [A]


@register("model", "ABMIL")
class ABMIL(_ABMILCore):
    """
    论文版（vanilla attention）的向量输入实现。
    用法：
        model = ABMIL(input_dim=D, M=500, L=128, attention_branches=1, num_classes=2, dropout=0.0)
        logits, att = model(x)  # x: [B,N,D]/[N,D]/list
    训练：
        - 二分类（C=2）：CrossEntropyLoss(logits, y.long())
        - 二分类（C=1）：BCEWithLogitsLoss(logits.squeeze(1), y.float())
    """
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
    """
    Gated Attention 版本：tanh(VH) ⊙ sigm(UH)
    """
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
