# src/models/dsmil.py
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class FCLayer(nn.Module):
    def __init__(self, in_size, out_size=1):
        super(FCLayer, self).__init__()
        self.fc = nn.Sequential(nn.Linear(in_size, out_size))
    def forward(self, feats):
        x = self.fc(feats)
        return x  # [N, out_size]

class IClassifier(nn.Module):
    def __init__(self, feature_extractor, feature_size, output_class):
        super(IClassifier, self).__init__()
        self.feature_extractor = feature_extractor      
        self.fc = nn.Linear(feature_size, output_class) 
    def forward(self, x):
        feats = self.feature_extractor(x)               
        if isinstance(feats, (tuple, list)):            
            feats = feats[0]
        c = self.fc(feats.view(feats.shape[0], -1))     
        return feats.view(feats.shape[0], -1), c        

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
        
        self.fcc = nn.Conv1d(output_class, output_class, kernel_size=input_size)

    def forward(self, feats, c): 
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
    def __init__(self, input_dim: int, num_classes: int = 2,
                 dropout: float = 0.0, nonlinear: bool = True, passing_v: bool = False, **kwargs):
        super().__init__()
        i_classifier = IClassifier(feature_extractor=nn.Identity(),
                                   feature_size=input_dim,
                                   output_class=num_classes)
        b_classifier = BClassifier(input_size=input_dim,
                                   output_class=num_classes,
                                   dropout_v=dropout,
                                   nonlinear=nonlinear,
                                   passing_v=passing_v)
        self.net = MILNet(i_classifier, b_classifier)

    def forward(self, x: torch.Tensor):
        classes, prediction_bag, A, B = self.net(x)

        return prediction_bag, {"inst_scores": classes, "attn": A, "bag_embed": B}
