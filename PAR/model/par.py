import math
from typing import List, Tuple, Optional
import torch
import torch.nn as nn


class CrossAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.h = num_heads
        self.dk = d_model // num_heads

        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.o = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x_q: torch.Tensor, x_kv: torch.Tensor, mask: Optional[torch.Tensor] = None,return_scores: bool = False):
        """
        x_q:  [B, Lq, D]
        x_kv: [B, Lk, D]
        mask: [B, 1, 1, Lk]  (True = keep, False = mask)
        """
        B, Lq, D = x_q.shape
        _, Lk, _ = x_kv.shape

        Q = self.q(x_q).view(B, Lq, self.h, self.dk).transpose(1, 2)  # [B,H,Lq,dk]
        K = self.k(x_kv).view(B, Lk, self.h, self.dk).transpose(1, 2)  # [B,H,Lk,dk]
        V = self.v(x_kv).view(B, Lk, self.h, self.dk).transpose(1, 2)  # [B,H,Lk,dk]

        scores = (Q @ K.transpose(-2, -1)) / math.sqrt(self.dk)        # [B,H,Lq,Lk]
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attn = torch.softmax(scores, dim=-1)
        attn = self.drop(attn)
        out = attn @ V                                                # [B,H,Lq,dk]
        out = out.transpose(1, 2).contiguous().view(B, Lq, D)         # [B,Lq,D]
        out = self.o(out)
        if return_scores:
            return out, attn, scores
        else:
            return out, attn


class CrossAttnBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.cross = CrossAttention(d_model, num_heads, dropout)
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )
        self.norm_ffn = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x_q: torch.Tensor, x_kv: torch.Tensor, mask: Optional[torch.Tensor] = None, return_scores: bool = False):
        q = self.norm_q(x_q)
        kv = self.norm_kv(x_kv)
        if return_scores:
            attn_out, attn_map,scores = self.cross(q, kv, mask=mask,return_scores=True)
        else:
            attn_out, attn_map = self.cross(q, kv, mask=mask)
            scores = None   
        x_q = x_q + self.drop(attn_out)

        f = self.ffn(self.norm_ffn(x_q))
        x_q = x_q + self.drop(f)
        return x_q, attn_map,scores


class CrossAttnStack(nn.Module):
    def __init__(self, d_model: int, num_heads: int, num_layers: int = 1, dropout: float = 0.1):
        super().__init__()
        assert num_layers >= 1
        self.layers = nn.ModuleList([
            CrossAttnBlock(d_model, num_heads, dropout) for _ in range(num_layers)
        ])

    def forward(self, x_q: torch.Tensor, x_kv: torch.Tensor, mask: Optional[torch.Tensor] = None,return_scores: bool = False):
        attn_all = []
        scores_last = None
        for layer in self.layers:
            if return_scores:
                x_q, attn, scores = layer(x_q, x_kv, mask,return_scores=True)
                scores_last = scores
            else:
                x_q, attn,scores_last = layer(x_q, x_kv, mask)
                 
            attn_all.append(attn)
        attn_last = attn_all[-1]
        return x_q, attn_last, scores_last



class SignatureTokenizer(nn.Module):
    """把 [B,Lr] 的 signature 分数编码成 [B,Lr,D] token；带每条通路的 id_embed"""

    def __init__(self, num_signatures: int, d_model: int, hidden: int = 0):
        super().__init__()
        self.num_signatures = num_signatures
        self.id_embed = nn.Embedding(num_signatures, d_model)

        if hidden and hidden > 0:
            self.val_mlp = nn.Sequential(
                nn.Linear(1, hidden),
                nn.GELU(),
                nn.Linear(hidden, d_model),
            )
        else:
            self.val_mlp = nn.Linear(1, d_model)

        self.val_scale = nn.Parameter(torch.tensor(1.0))

        nn.init.xavier_uniform_(self.id_embed.weight)
        if isinstance(self.val_mlp, nn.Sequential):
            nn.init.xavier_uniform_(self.val_mlp[-1].weight)
        else:
            nn.init.xavier_uniform_(self.val_mlp.weight)

    def forward(self, sig_values: torch.Tensor, sig_mask: Optional[torch.Tensor] = None):
        """
        sig_values: [B,Lr]
        sig_mask:   [B,Lr] or None
        """
        B, Lr = sig_values.shape
        device = sig_values.device

        ids = torch.arange(Lr, device=device).unsqueeze(0).expand(B, Lr)
        id_tok = self.id_embed(ids)                       # [B,Lr,D]
        val_tok = self.val_mlp(sig_values.unsqueeze(-1))  # [B,Lr,D]
        tokens = id_tok + self.val_scale * val_tok        # [B,Lr,D]

        if sig_mask is None:
            attn_mask = torch.ones(B, 1, 1, Lr, device=device, dtype=torch.bool)
        else:
            attn_mask = sig_mask.to(torch.bool).view(B, 1, 1, Lr)

        return tokens, attn_mask

class SignatureDecoderFromWSI(nn.Module):

    def __init__(
        self,
        d_model: int,
        num_signatures: int,
        num_heads: int = 6,
        dropout: float = 0.1,
        num_layers: int = 1,
    ):
        super().__init__()
        self.num_signatures = num_signatures
        self.cross_stack = CrossAttnStack(d_model, num_heads, num_layers=num_layers, dropout=dropout)
        self.readout = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, sig_queries_embed: torch.Tensor, x_wsi: torch.Tensor,
                wsi_mask: Optional[torch.Tensor] = None,return_scores: bool = False):
        """
        sig_queries_embed: [B,Lr,D] (E_sig)
        x_wsi:             [B,Lw,D]
        wsi_mask:          [B,Lw] or None
        """
        if wsi_mask is not None:
            mask = wsi_mask.view(x_wsi.size(0), 1, 1, -1)   # [B,1,1,Lw]
        else:
            mask = None

        ctx, attn_last, scores = self.cross_stack(sig_queries_embed, x_wsi, mask=mask,return_scores = return_scores)  # [B,Lr,D]
        pred = self.readout(ctx).squeeze(-1)                                       # [B,Lr]
        predlogits = ctx
        return pred, attn_last,scores,predlogits


from src.utils import registry

@registry.register("model", "PAR")
class PAR(nn.Module):
    """
      xs: list，长度 B，每个元素是 feats 或 (feats, coords)，feats:[N_i, input_dim]
      sig_values: [B, num_signatures]  (可选)

      - "stage1"        : 有 RNA signature，用来做 WSI↔signature 对齐 + 预测 16 维 signature，
                          此时 forward 返回 logits.shape=[B, num_signatures]。
      - "stage2_pseudo" : 无 RNA（或不传 sig_values），用 E_sig 从 WSI 解码出 pseudo signature，
                          再 + 残差头预测 immunescore，logits.shape=[B,1]。
    """

    def __init__(
        self,
        input_dim: int = 1536,
        num_signatures: int = 16,
        d_model: int = 512,
        num_heads: int = 8,
        dropout: float = 0.1,
        hidden: int = 0,
        num_xattn_layers: int = 3,   # 正向 WSI×RNA cross-attn 层数
        num_sigdec_layers: int = 3,  # 反向 E_sig×WSI 解码层数
    ):
        super().__init__()
        self.current_stage = "stage1"
        self.num_signatures = num_signatures
        

        # 1) WSI 特征映射：1536 → d_model
        self.patch_proj = nn.Linear(input_dim, d_model)

        # 2) Signature tokenizer
        self.sig_tokenizer = SignatureTokenizer(num_signatures, d_model, hidden=hidden)

        # 3) 正向：WSI(Q) × RNA/signature(K,V)
        self.cross_attention = CrossAttnStack(d_model, num_heads, num_layers=num_xattn_layers, dropout=dropout)

        # 4) 反向解码器：E_sig→WSI
        self.sig_decoder = SignatureDecoderFromWSI(
            d_model,
            num_signatures,
            num_heads=num_heads,
            dropout=dropout,
            num_layers=num_sigdec_layers,
        )
        
        self.forward_head = nn.Sequential(
            nn.Linear(d_model, d_model//2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model//2, 1),
        )
        


    def set_stage(self, stage: str):
        assert stage in {"stage1", "stage1b","stage2_pseudo"}
        self.current_stage = stage

    # ---- 内部：把 list[(N_i,D)] pad 成 [B,L_max,D] ----

    def _prepare_wsi_batch(self, xs: List):
        """
        xs: list，长度 B，每个元素是 feats 或 (feats, coords)
        返回:
          x_wsi:   [B,L_max,Din]
          wsi_mask:[B,L_max] (True=有效)
        """
        device = next(self.parameters()).device

        if isinstance(xs, torch.Tensor):
            x = xs.to(device, dtype=torch.float32)
            if x.dim() == 2:
                x = x.unsqueeze(0)
            B, L, D = x.shape
            mask = torch.ones(B, L, dtype=torch.bool, device=device)
            return x, mask

        assert isinstance(xs, list), "xs 必须是 list 或 Tensor"
        feats_list = []
        lens = []
        for item in xs:
            if isinstance(item, (list, tuple)):
                feats, _coords = item
            else:
                feats = item
            feats = torch.as_tensor(feats, dtype=torch.float32, device=device)
            feats_list.append(feats)
            lens.append(feats.shape[0])

        B = len(feats_list)
        D = feats_list[0].shape[1]
        L_max = max(lens)

        x_batch = torch.zeros(B, L_max, D, dtype=torch.float32, device=device)
        mask = torch.zeros(B, L_max, dtype=torch.bool, device=device)

        for i, f in enumerate(feats_list):
            L = f.shape[0]
            x_batch[i, :L, :] = f
            mask[i, :L] = True

        return x_batch, mask

    # ---- forward ----

    def forward(
        self,
        xs,
        sig_values: Optional[torch.Tensor] = None,
        stage: Optional[str] = None,
    ):
        
        if stage is not None:
            self.set_stage(stage)

        device = next(self.parameters()).device

      
        x_raw, wsi_mask = self._prepare_wsi_batch(xs)          # [B,Lw,input_dim], [B,Lw]
        x_wsi = self.patch_proj(x_raw)                         # [B,Lw,D]

        B, Lw, D = x_wsi.shape

        extras = {
            "stage": self.current_stage,
            "attn_wsi_rna": None,
            "sig_from_z_wsi": None,     
            "sig_from_decoder": None,   
            "sigdec_attn": None,
            "patch_weights": None,
            "immune_from_sig": None,
            "immune_residual": None,
            "sigdec_scores": None,
        }

        
        ids = torch.arange(self.num_signatures, device=device).unsqueeze(0).expand(B, -1)
        Esig = self.sig_tokenizer.id_embed(ids)                # [B,Lr,D]

        # ---------- Stage1：有 RNA signature，训练对齐 + 预测 16 维 ----------

        if self.current_stage in {"stage1", "stage1b"}:
            assert sig_values is not None, "stage1/1b 需要传入 sig_values（[B, num_signatures]）"
            sig_values = sig_values.to(device)
            x_rna, rna_mask = self.sig_tokenizer(sig_values, None)   # [B,Lr,D], [B,1,1,Lr]

            # 正向 cross-attn: WSI(Q) × RNA(K,V)
            z_wsi, cross_map, _ = self.cross_attention(x_wsi, x_rna, mask=rna_mask,return_scores = True)
            extras["attn_wsi_rna"] = cross_map.detach()

            # 反向解码：Esig(Q) × WSI(K,V) → signature 预测
            sig_pred, sigdec_attn,sigdec_scores,sig_token = self.sig_decoder(Esig, x_wsi, wsi_mask=wsi_mask,return_scores = True) # [B,num_signature], attn: [B, H, num_signature, patch]
            extras["sig_from_decoder"] = sig_pred.detach()              # [B,num_signatures]
            extras["sigdec_attn"] = sigdec_attn.detach()
            extras["sigdec_scores"] = sigdec_scores.detach()
    
            if self.current_stage == "stage1":
                # Stage1A：只用反向头，保持和旧版本兼容
                logits = sig_pred
                return logits, extras
            
            #对 heads 求平均 → [B, num_signatures, Lw]
            W = sigdec_attn.mean(dim=1)
            H_sig = torch.bmm(W, z_wsi)              # [B, num_signatures, D]

            B, S, D = H_sig.shape
            pred_fwd = self.forward_head(H_sig).view(B, S)  # [B, num_signatures]
            # Stage1B：同时输出反向 & 正向两条预测
            pred_bwd = sig_pred                             # [B,num_signatures]
            
            extras["sig_from_z_wsi"] = pred_fwd.detach()

            return (pred_bwd, pred_fwd), extras

        # ---------- Stage2----------

        # 2A) 反向 pseudo signature（可选）
        if self.current_stage == "stage2_pseudo":
            sig_pred, sigdec_attn,sigdec_scores,sig_token = self.sig_decoder(Esig, x_wsi, wsi_mask=wsi_mask,return_scores = True)
            extras["sig_from_decoder"] = sig_pred.detach()
            extras["sigdec_attn"] = sigdec_attn.detach()
            extras["sigdec_scores"] = sigdec_scores.detach()
            
            pred = sig_pred
            
        else:
            raise ValueError(f"Unknown stage={self.current_stage}")
        return pred, extras, sig_token
        

