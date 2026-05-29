# src/models/rimil_mss_hot.py
from typing import Optional, Dict, Any
from typing import Optional, List, Tuple
import torch
import torch.nn as nn

from src.models.par import PAR   
from src.utils import registry

class GatedABMILPooling(nn.Module):
    """
    Gated attention MIL pooling (Ilse et al., 2018 style).
    """
    def __init__(self, d_model: int, attn_hidden: int = 256, dropout: float = 0.0):
        super().__init__()
        self.V = nn.Linear(d_model, attn_hidden)
        self.U = nn.Linear(d_model, attn_hidden)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.drop = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()
        self.w = nn.Linear(attn_hidden, 1)

    def forward(self, ctx: torch.Tensor, mask: Optional[torch.Tensor] = None):
        """
        ctx:  [B, S, D]
        mask: [B, S] optional
        """
        v = self.tanh(self.V(ctx))          # [B,S,H]
        u = self.sigmoid(self.U(ctx))       # [B,S,H]
        a = self.drop(v * u)                # [B,S,H]
        scores = self.w(a).squeeze(-1)      # [B,S]

        if mask is not None:
            scores = scores.masked_fill(~mask, -1e9)

        attn = torch.softmax(scores, dim=1) # [B,S]
        z_bag = torch.sum(ctx * attn.unsqueeze(-1), dim=1)  # [B,D]
        return z_bag, attn
class SigGatedMILClassifier(nn.Module):
    def __init__(self, d_model: int, hidden: int = 64, dropout: float = 0.1, attn_hidden: int = 256, attn_dropout: float = 0.0):
        super().__init__()
        self.pool = GatedABMILPooling(d_model=d_model, attn_hidden=attn_hidden, dropout=attn_dropout)
        self.clf = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, ctx: torch.Tensor, mask: Optional[torch.Tensor] = None):
        z_bag, w = self.pool(ctx, mask=mask)
        logits = self.clf(z_bag)
        return logits, w

class PathwayMLPHead(nn.Module):
    def __init__(self, d_model, hidden=512, dropout=0.1, num_classes=1,num_signatures=16):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)               # 对每个token做LN更稳
        self.mlp = nn.Sequential(
            nn.Linear(num_signatures*d_model, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes)
        )

    def forward(self, x):  # x: [B,16,D]
        x = self.norm(x)                           # still [B,16,D]
        x = x.reshape(x.size(0), -1)               # [B,16*D]
        return self.mlp(x)
    
@registry.register("model", "PAR_TA")
class PAR_TA(nn.Module):
    

    def __init__(
        self,
        input_dim: int = 1536,
        num_signatures: int = 16,
        d_model: int = 512,
        num_heads: int = 8,
        dropout: float = 0.1,
        hidden: int = 0,
        num_xattn_layers: int = 3,
        num_sigdec_layers: int = 3,
        

        
        pretrained_path: str = "",
        freeze_backbone: bool = True,

    ):
        super().__init__()

        
        self.backbone = PAR(
            input_dim=input_dim,
            num_signatures=num_signatures,
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
            hidden=hidden,
            num_xattn_layers=num_xattn_layers,
            num_sigdec_layers=num_sigdec_layers,
            
        )

        
        if pretrained_path:
            ckpt = torch.load(pretrained_path, map_location="cpu")
            state_dict = ckpt.get("model", ckpt)
            missing, unexpected = self.backbone.load_state_dict(state_dict, strict=False)
            print(f"[PAR_TA] Loaded pretrained backbone from: {pretrained_path}")
            if missing:
                print("  missing keys:", missing)
            if unexpected:
                print("  unexpected keys:", unexpected)

        
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
            print("[PAR_TA] Backbone is frozen (feature extractor only).")
        
        self.mil = SigGatedMILClassifier(d_model)

        self.num_signatures = num_signatures

    def forward(
        self,
        xs,
        sig_values: Optional[torch.Tensor] = None,
        stage: Optional[str] = None,
    ):
        
        
        _stage = stage if stage is not None else "stage2_pseudo"

        
        pred, extras,sig_token = self.backbone(xs, sig_values=None, stage=_stage)

        if extras.get("sig_from_decoder", None) is None:
            raise RuntimeError(
                "Backbone did not produce 'sig_from_decoder'. "
                "Please make sure stage='stage2_pseudo' and the ckpt is trained with decoder."
            )

        bag_pred, w = self.mil(sig_token)
        return bag_pred, w


@registry.register("model", "PAR_TA_B")
class PAR_TA_B(nn.Module):
    

    def __init__(
        self,
        input_dim: int = 1536,
        num_signatures: int = 16,
        d_model: int = 512,
        num_heads: int = 8,
        dropout: float = 0.1,
        hidden: int = 0,
        num_xattn_layers: int = 3,
        num_sigdec_layers: int = 3,

        pretrained_path: str = "",
        freeze_backbone: bool = True,

       
        # : IFNG,APM,MHCII,TLS,NK,Treg,M1,M2,Bcell,Proliferation,EMT,TGFb,Endothelial,Stromal,CD8,CYT
        # B = Treg, M2, Proliferation, EMT, TGFb, Endothelial, Stromal
        selected_sig_idx=(5, 7, 9, 10, 11, 12, 13),
    ):
        super().__init__()

        # 1) backbone
        self.backbone = PAR(
            input_dim=input_dim,
            num_signatures=num_signatures,
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
            hidden=hidden,
            num_xattn_layers=num_xattn_layers,
            num_sigdec_layers=num_sigdec_layers,
        )

        # 2) load pretrained
        if pretrained_path:
            ckpt = torch.load(pretrained_path, map_location="cpu")
            state_dict = ckpt.get("model", ckpt)
            missing, unexpected = self.backbone.load_state_dict(state_dict, strict=False)
            print(f"[PAR_TA_B] Loaded pretrained backbone from: {pretrained_path}")
            if missing:
                print("  missing keys:", missing)
            if unexpected:
                print("  unexpected keys:", unexpected)

        # 3) freeze backbone
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
            print("[PAR_TA_B] Backbone is frozen (feature extractor only).")

        self.num_signatures = num_signatures
        self.selected_sig_idx = tuple(selected_sig_idx)

        
        self.mil = SigGatedMILClassifier(d_model)

    def forward(
        self,
        xs,
        sig_values: Optional[torch.Tensor] = None,
        stage: Optional[str] = None,
    ):
        _stage = stage if stage is not None else "stage2_pseudo"

        pred, extras, sig_token = self.backbone(xs, sig_values=None, stage=_stage)

        if sig_token is None:
            raise RuntimeError(
                "Backbone did not return sig_token. "
                "Please make sure stage='stage2_pseudo' and the decoder is enabled."
            )

        # sig_token: [B, num_signatures, d_model]
        idx = torch.as_tensor(self.selected_sig_idx, device=sig_token.device, dtype=torch.long)
        sig_token_sel = sig_token.index_select(dim=1, index=idx)  # [B, k, d_model]

        bag_pred, w = self.mil(sig_token_sel)

        if extras is None:
            extras = {}
        extras["sig_token_selected"] = sig_token_sel
        extras["pred_selected_idx"] = idx

        return bag_pred, w

@registry.register("model", "PAR_TA_Acore")
class PAR_TA_Acore(nn.Module):

    def __init__(
        self,
        input_dim: int = 1536,
        num_signatures: int = 16,
        d_model: int = 512,
        num_heads: int = 8,
        dropout: float = 0.1,
        hidden: int = 0,
        num_xattn_layers: int = 3,
        num_sigdec_layers: int = 3,

        pretrained_path: str = "",
        freeze_backbone: bool = True,

        # : IFNG,APM,MHCII,TLS,NK,Treg,M1,M2,Bcell,Proliferation,EMT,TGFb,Endothelial,Stromal,CD8,CYT
        # B = Treg, M2, Proliferation, EMT, TGFb, Endothelial, Stromal
        selected_sig_idx=(0,1,2,3,4,6,8,14,15),
    ):
        super().__init__()

        # 1) backbone
        self.backbone = PAR(
            input_dim=input_dim,
            num_signatures=num_signatures,
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
            hidden=hidden,
            num_xattn_layers=num_xattn_layers,
            num_sigdec_layers=num_sigdec_layers,
        )

        # 2) load pretrained
        if pretrained_path:
            ckpt = torch.load(pretrained_path, map_location="cpu")
            state_dict = ckpt.get("model", ckpt)
            missing, unexpected = self.backbone.load_state_dict(state_dict, strict=False)
            print(f"[PAR_TA_Acore] Loaded pretrained backbone from: {pretrained_path}")
            if missing:
                print("  missing keys:", missing)
            if unexpected:
                print("  unexpected keys:", unexpected)

        # 3) freeze backbone
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
            print("[PAR_TA_Acore] Backbone is frozen (feature extractor only).")

        self.num_signatures = num_signatures
        self.selected_sig_idx = tuple(selected_sig_idx)

        # MIL 头保持不变，输入仍是 d_model
        self.mil = SigGatedMILClassifier(d_model)

    def forward(
        self,
        xs,
        sig_values: Optional[torch.Tensor] = None,
        stage: Optional[str] = None,
    ):
        _stage = stage if stage is not None else "stage2_pseudo"

        pred, extras, sig_token = self.backbone(xs, sig_values=None, stage=_stage)

        if sig_token is None:
            raise RuntimeError(
                "Backbone did not return sig_token. "
                "Please make sure stage='stage2_pseudo' and the decoder is enabled."
            )

        # sig_token: [B, num_signatures, d_model]
        idx = torch.as_tensor(self.selected_sig_idx, device=sig_token.device, dtype=torch.long)
        sig_token_sel = sig_token.index_select(dim=1, index=idx)  # [B, k, d_model]

        bag_pred, w = self.mil(sig_token_sel)

        if extras is None:
            extras = {}
        extras["sig_token_selected"] = sig_token_sel
        extras["pred_selected_idx"] = idx

        return bag_pred, w


@registry.register("model", "PAR_MLP")
class PAR_MLP(nn.Module):

    def __init__(
        self,
        input_dim: int = 1536,
        num_signatures: int = 16,
        d_model: int = 512,
        num_heads: int = 8,
        dropout: float = 0.1,
        hidden: int = 0,
        num_xattn_layers: int = 3,
        num_sigdec_layers: int = 3,

        
        pretrained_path: str = "",
        freeze_backbone: bool = True,
        num_classes:int =1,
        cls_hidden_dim = 32,

    ):
        super().__init__()

        
        self.backbone = PAR(
            input_dim=input_dim,
            num_signatures=num_signatures,
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
            hidden=hidden,
            num_xattn_layers=num_xattn_layers,
            num_sigdec_layers=num_sigdec_layers,
            
        )

        
        if pretrained_path:
            ckpt = torch.load(pretrained_path, map_location="cpu")
            
            state_dict = ckpt.get("model", ckpt)
            missing, unexpected = self.backbone.load_state_dict(state_dict, strict=False)
            print(f"[PAR_MLP] Loaded pretrained backbone from: {pretrained_path}")
            if missing:
                print("  missing keys:", missing)
            if unexpected:
                print("  unexpected keys:", unexpected)

        
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
            print("[PAR_MLP] Backbone is frozen (feature extractor only).")

        self.num_signatures = num_signatures
        self.cls_head = nn.Sequential(
            nn.Linear(num_signatures, cls_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(cls_hidden_dim, num_classes),
        )

    def forward(
        self,
        xs,
        sig_values: Optional[torch.Tensor] = None,
        stage: Optional[str] = None,
    ):
        
        _stage = stage if stage is not None else "stage2_pseudo"

        
        pred, extras,sig_token = self.backbone(xs, sig_values=None, stage=_stage)

        if extras.get("sig_from_decoder", None) is None:
            raise RuntimeError(
                "Backbone did not produce 'sig_from_decoder'. "
                "Please make sure stage='stage2_pseudo' and the ckpt is trained with decoder."
            )

        logit = self.cls_head(pred)
        return logit,extras


