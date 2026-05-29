# src/models/dsmil.py
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

# ========== 你给的 DSMIL 源码（原样，只有极小的健壮性兼容） ==========
class FCLayer(nn.Module):
    def __init__(self, in_size, out_size=1):
        super(FCLayer, self).__init__()
        self.fc = nn.Sequential(nn.Linear(in_size, out_size))
    def forward(self, feats):
        x = self.fc(feats)
        # 原论文代码里有些实现会返回 (feats, x)。为兼容 IClassifier，这里只返回变换后的 feats。
        return x  # [N, out_size]

class IClassifier(nn.Module):
    def __init__(self, feature_extractor, feature_size, output_class):
        super(IClassifier, self).__init__()
        self.feature_extractor = feature_extractor      # 期望: (N,K) -> (N,K')
        self.fc = nn.Linear(feature_size, output_class) # K' -> C
    def forward(self, x):
        feats = self.feature_extractor(x)               # [N, K']
        if isinstance(feats, (tuple, list)):            # 兼容极少数返回 (feats, aux) 的实现
            feats = feats[0]
        c = self.fc(feats.view(feats.shape[0], -1))     # [N, C]
        return feats.view(feats.shape[0], -1), c        # (N,K'), (N,C)

class BClassifier(nn.Module):
    def __init__(self, input_size, output_class, dropout_v=0.0, nonlinear=True, passing_v=False): # K, C
        super(BClassifier, self).__init__()
        if nonlinear:
            self.q = nn.Sequential(nn.Linear(input_size, 128), nn.ReLU(), nn.Linear(128, 128), nn.Tanh())
        else:
            self.q = nn.Linear(input_size, 128)
        if passing_v:
            self.v = nn.Sequential(
                nn.Dropout(dropout_v),
                nn.Linear(input_size, input_size),
                nn.ReLU()
            )
        else:
            self.v = nn.Identity()
        # 1D 卷积用于多类别（含二分类）
        self.fcc = nn.Conv1d(output_class, output_class, kernel_size=input_size)

    def forward(self, feats, c): # feats: [N, K], c: [N, C]
        device = feats.device
        V = self.v(feats)                        # [N, K]
        Q = self.q(feats).view(feats.shape[0], -1)  # [N, Q]

        _, m_indices = torch.sort(c, 0, descending=True)       # [N, C]
        m_feats = torch.index_select(feats, dim=0, index=m_indices[0, :])  # [C, K]
        q_max = self.q(m_feats)                                 # [C, Q]
        A = torch.mm(Q, q_max.transpose(0, 1))                  # [N, C]
        A = F.softmax(A / torch.sqrt(torch.tensor(Q.shape[1], dtype=torch.float32, device=device)), dim=0)
        B = torch.mm(A.transpose(0, 1), V)                      # [C, K]
        B = B.view(1, B.shape[0], B.shape[1])                   # [1, C, K]
        C = self.fcc(B).view(1, -1)                             # [1, C]
        return C, A, B

class MILNet(nn.Module):
    def __init__(self, i_classifier, b_classifier):
        super(MILNet, self).__init__()
        self.i_classifier = i_classifier
        self.b_classifier = b_classifier
    def forward(self, x):
        feats, classes = self.i_classifier(x)          # (N,K), (N,C)
        prediction_bag, A, B = self.b_classifier(feats, classes) # [1,C], [N,C], [1,C,K]
        return classes, prediction_bag, A, B

# ========== 薄包装：适配你的通用 trainer（只关心 bag-level logits） ==========
try:
    from src.utils import registry
except Exception:
    class _Reg:
        def register(self, *args, **kwargs):
            def deco(cls): return cls
            return deco
    registry = _Reg()

@registry.register("model", "DSMIL")
class DSMIL(nn.Module):
    """
    统一构造参数（对齐你的 smart_build_model）：
      - input_dim: 特征维度 K
      - num_classes: 类别数 C（或 n_classes）
      - dropout: 传给 BClassifier 的 dropout_v（当 passing_v=True 时生效）
      - nonlinear: BClassifier 的 q 是否非线性
      - passing_v: 是否使用 v MLP（True 则对 V 做一层映射）
    forward(x):
      - x: [N, K]（单个 bag）
      - 返回: logits [1, C]    （trainer 会自动据形状选择 CE/BCE）
    """
    def __init__(self, input_dim: int, num_classes: int = 2,
                 dropout: float = 0.0, nonlinear: bool = True, passing_v: bool = False, **kwargs):
        super().__init__()
        # 实例级：直接用 Identity，当作特征已抽取好的 [N,K]
        i_classifier = IClassifier(feature_extractor=nn.Identity(),
                                   feature_size=input_dim,
                                   output_class=num_classes)
        # bag 级
        b_classifier = BClassifier(input_size=input_dim,
                                   output_class=num_classes,
                                   dropout_v=dropout,
                                   nonlinear=nonlinear,
                                   passing_v=passing_v)
        self.net = MILNet(i_classifier, b_classifier)

    def forward(self, x: torch.Tensor):
        # x: [N, K]
        classes, prediction_bag, A, B = self.net(x)
        # 只返回 bag-level logits 给 trainer（trainer 会只取第一个返回值）
        return prediction_bag, {"inst_scores": classes, "attn": A, "bag_embed": B}
