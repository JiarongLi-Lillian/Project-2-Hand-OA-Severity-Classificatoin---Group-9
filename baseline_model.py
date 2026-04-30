import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F


class OrdinalRegressionModule(nn.Module):
    def __init__(self, num_classes: int = 5):
        super().__init__()
        self.num_classes = num_classes
        self.cut_points = nn.Parameter(
            torch.arange(num_classes - 1, dtype=torch.float32)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cut_points = torch.sort(self.cut_points)[0]
        cum_probs  = torch.sigmoid(cut_points.unsqueeze(0) - x)
        zeros  = torch.zeros(x.size(0), 1, device=x.device)
        ones   = torch.ones(x.size(0),  1, device=x.device)
        padded = torch.cat([zeros, cum_probs, ones], dim=1)
        probs  = padded[:, 1:] - padded[:, :-1]
        return probs


class _ORMBackbone(nn.Module):
    def __init__(self, backbone: nn.Module, orm: OrdinalRegressionModule):
        super().__init__()
        self.backbone = backbone
        self.orm      = orm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scalar = self.backbone(x)
        probs  = self.orm(scalar)
        return probs


class CORALLayer(nn.Module):
    def __init__(self, num_classes: int = 5):
        super().__init__()
        self.num_classes = num_classes
        self.bias = nn.Parameter(
            torch.arange(num_classes - 2, -1, -1, dtype=torch.float32)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.bias.unsqueeze(0)


class _CORALBackbone(nn.Module):
    def __init__(self, backbone: nn.Module, coral: CORALLayer):
        super().__init__()
        self.backbone = backbone
        self.coral    = coral

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scalar = self.backbone(x)
        logits = self.coral(scalar)
        return logits


def coral_predict(logits: torch.Tensor) -> torch.Tensor:
    probs  = torch.sigmoid(logits)
    binary = (probs > 0.5).long()
    return binary.sum(dim=1)


def coral_probs(logits: torch.Tensor, num_classes: int = 5) -> torch.Tensor:
    p     = torch.sigmoid(logits)
    zeros = torch.zeros(p.size(0), 1, device=p.device)
    ones  = torch.ones(p.size(0),  1, device=p.device)
    cum   = torch.cat([ones, p, zeros], dim=1)
    probs = cum[:, :-1] - cum[:, 1:]
    return probs.clamp(min=0)

def corn_predict(logits: torch.Tensor) -> torch.Tensor:
    probs     = torch.sigmoid(logits)
    cum_probs = torch.cumprod(probs, dim=1)
    return (cum_probs > 0.5).sum(dim=1)


def corn_probs(logits: torch.Tensor, num_classes: int = 5) -> torch.Tensor:
    probs     = torch.sigmoid(logits)
    cum_probs = torch.cumprod(probs, dim=1)
    zeros     = torch.zeros(cum_probs.size(0), 1, device=logits.device)
    ones      = torch.ones(cum_probs.size(0),  1, device=logits.device)
    cum       = torch.cat([ones, cum_probs, zeros], dim=1)
    return (cum[:, :-1] - cum[:, 1:]).clamp(min=0)


class ChannelAttention(nn.Module):
    def __init__(self, in_channels: int, reduction_ratio: int = 16):
        super().__init__()
        reduced = max(in_channels // reduction_ratio, 8)
        self.shared_mlp = nn.Sequential(
            nn.Linear(in_channels, reduced, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced, in_channels, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = x.mean(dim=[2, 3])
        mx  = x.amax(dim=[2, 3])
        att = torch.sigmoid(self.shared_mlp(avg) + self.shared_mlp(mx))
        return x * att.unsqueeze(2).unsqueeze(3)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = x.mean(dim=1, keepdim=True)
        mx  = x.amax(dim=1, keepdim=True)
        cat = torch.cat([avg, mx], dim=1)
        att = torch.sigmoid(self.conv(cat))
        return x * att


class CBAM(nn.Module):
    def __init__(self, in_channels: int,
                 reduction_ratio: int = 16,
                 spatial_kernel: int = 7):
        super().__init__()
        self.channel = ChannelAttention(in_channels, reduction_ratio)
        self.spatial = SpatialAttention(spatial_kernel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel(x)
        x = self.spatial(x)
        return x


class MobileNetV2WithCBAM(nn.Module):
    def __init__(self, num_classes: int = 5):
        super().__init__()
        base = models.mobilenet_v2(weights='IMAGENET1K_V1')
        self.features   = base.features
        self.cbam       = CBAM(in_channels=1280, reduction_ratio=16)
        self.pool       = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(base.classifier[1].in_features, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.cbam(x)
        x = self.pool(x).flatten(1)
        return self.classifier(x)


class MobileNetV2WithMSFM(nn.Module):
    def __init__(self, num_classes: int = 5, proj_dim: int = 256):
        super().__init__()
        base = models.mobilenet_v2(weights='IMAGENET1K_V1')

        self.low_features  = base.features[:14]
        self.high_features = base.features[14:]

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.proj_low  = nn.Sequential(
            nn.Conv2d(96,   proj_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(proj_dim),
            nn.ReLU(inplace=True),
        )
        self.proj_high = nn.Sequential(
            nn.Conv2d(1280, proj_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(proj_dim),
            nn.ReLU(inplace=True),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(proj_dim * 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f_low  = self.low_features(x)
        f_high = self.high_features(f_low)
        f_low  = self.pool(self.proj_low(f_low)).flatten(1)
        f_high = self.pool(self.proj_high(f_high)).flatten(1)
        fused = torch.cat([f_low, f_high], dim=1)
        return self.classifier(fused)

class MobileNetV2WithMSFMCORN(nn.Module):
    def __init__(self, num_classes: int = 5, proj_dim: int = 256):
        super().__init__()
        base = models.mobilenet_v2(weights='IMAGENET1K_V1')

        self.low_features  = base.features[:14]
        self.high_features = base.features[14:]

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.proj_low  = nn.Sequential(
            nn.Conv2d(96,   proj_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(proj_dim),
            nn.ReLU(inplace=True),
        )
        self.proj_high = nn.Sequential(
            nn.Conv2d(1280, proj_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(proj_dim),
            nn.ReLU(inplace=True),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(proj_dim * 2, num_classes - 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f_low  = self.low_features(x)
        f_high = self.high_features(f_low)
        f_low  = self.pool(self.proj_low(f_low)).flatten(1)
        f_high = self.pool(self.proj_high(f_high)).flatten(1)
        fused  = torch.cat([f_low, f_high], dim=1)
        return self.classifier(fused)

class MobileNetV2CORALWide(nn.Module):
    def __init__(self, num_classes: int = 5, hidden_dim: int = 64):
        super().__init__()
        base = models.mobilenet_v2(weights='IMAGENET1K_V1')
        self.features = base.features
        self.pool     = nn.AdaptiveAvgPool2d(1)

        self.shared = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(1280, hidden_dim),
            nn.ReLU(inplace=True),
        )

        self.fc_weight = nn.Linear(hidden_dim, 1, bias=False)
        self.bias = nn.Parameter(
            torch.arange(num_classes - 2, -1, -1, dtype=torch.float32)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x      = self.features(x)
        x      = self.pool(x).flatten(1)
        x      = self.shared(x)
        x      = self.fc_weight(x)
        logits = x + self.bias.unsqueeze(0)
        return logits

class MobileNetV2WithMSFMOrdinal(nn.Module):
    def __init__(self, num_classes: int = 5, proj_dim: int = 256):
        super().__init__()
        base = models.mobilenet_v2(weights='IMAGENET1K_V1')

        self.low_features  = base.features[:14]
        self.high_features = base.features[14:]
        self.pool = nn.AdaptiveAvgPool2d(1)

        self.proj_low  = nn.Sequential(
            nn.Conv2d(96,   proj_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(proj_dim), nn.ReLU(inplace=True),
        )
        self.proj_high = nn.Sequential(
            nn.Conv2d(1280, proj_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(proj_dim), nn.ReLU(inplace=True),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(proj_dim * 2, num_classes),
        )

        self.ordinal_proj = nn.Linear(num_classes, 1, bias=False)
        self.ordinal_bias = nn.Parameter(
            torch.arange(num_classes - 2, -1, -1, dtype=torch.float32)
        )

    def forward(self, x):
        f_low  = self.low_features(x)
        f_high = self.high_features(f_low)
        f_low  = self.pool(self.proj_low(f_low)).flatten(1)
        f_high = self.pool(self.proj_high(f_high)).flatten(1)
        fused  = torch.cat([f_low, f_high], dim=1)

        main_logits = self.classifier(fused)
        scalar      = self.ordinal_proj(main_logits)
        ord_logits  = scalar + self.ordinal_bias.unsqueeze(0)

        ord_probs   = coral_probs(ord_logits, num_classes=5)
        
        return ord_probs

class Conv1x1(nn.Module):
    """1x1 conv + BN + ReLU projection block."""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class LightCrossAttention(nn.Module):
    """
    Single-direction cross-attention:
      Q = KL backbone feature  (main task queries auxiliary for useful context)
      K,V = auxiliary task embeddings  [e_jsn, e_op, e_er]  (B, n_aux, kv_dim)

    Parameter count:
      q_proj: query_dim * kv_dim  = 512 * 64 = 32,768
      k_proj: kv_dim   * kv_dim  =  64 * 64 =  4,096
      v_proj: kv_dim   * kv_dim  =  64 * 64 =  4,096
      LayerNorm: 2 * kv_dim      =         128
      Total: ~41,088 parameters
    """
    def __init__(self, query_dim: int = 512, kv_dim: int = 64):
        super().__init__()
        self.q_proj = nn.Linear(query_dim, kv_dim, bias=False)
        self.k_proj = nn.Linear(kv_dim,    kv_dim, bias=False)
        self.v_proj = nn.Linear(kv_dim,    kv_dim, bias=False)
        self.scale  = kv_dim ** -0.5
        self.norm   = nn.LayerNorm(kv_dim)

    def forward(self, query: torch.Tensor, kv_stack: torch.Tensor) -> torch.Tensor:
        """
        Args:
            query:    (B, query_dim)   – MSFM KL feature
            kv_stack: (B, n_aux, kv_dim) – stacked auxiliary embeddings
        Returns:
            out: (B, kv_dim)  – attention-weighted auxiliary context
        """
        q = self.q_proj(query).unsqueeze(1)    # (B, 1,     kv_dim)
        k = self.k_proj(kv_stack)              # (B, n_aux, kv_dim)
        v = self.v_proj(kv_stack)              # (B, n_aux, kv_dim)

        # Scaled dot-product attention
        attn_w = torch.softmax(
            torch.bmm(q, k.transpose(1, 2)) * self.scale,
            dim=-1
        )                                      # (B, 1, n_aux)

        # Weighted sum over auxiliary sequence
        out = torch.bmm(attn_w, v).squeeze(1)  # (B, kv_dim)
        return self.norm(out)

class MobileNetV2MSFMMultitaskCrossAttn(nn.Module):
    def __init__(self,
                 proj_dim: int = 256,
                 task_emb_dim: int = 64,
                 num_kl: int = 5,
                 num_jsn: int = 4,
                 num_op: int = 4,
                 num_er: int = 2):
        super().__init__()

        # ── Backbone: MSFM two-branch split ──────────────────────────
        base = models.mobilenet_v2(weights='IMAGENET1K_V1')
        self.low_features  = base.features[:14]   # output: 96ch
        self.high_features = base.features[14:]   # output: 1280ch
        self.pool = nn.AdaptiveAvgPool2d(1)

        self.proj_low  = Conv1x1(96,   proj_dim)  # 96   → 256
        self.proj_high = Conv1x1(1280, proj_dim)  # 1280 → 256

        # ── Task-aware encoders (same routing as MobileNetV2MSFMMultitask) ──
        # JSN (corr=0.71): needs fine detail + semantic context → cat[low, high]
        self.jsn_enc = nn.Sequential(
            nn.Linear(proj_dim * 2, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, task_emb_dim),
            nn.ReLU(inplace=True),
        )
        # OP (corr=0.73): structural change → high-level only
        self.op_enc = nn.Sequential(
            nn.Linear(proj_dim, task_emb_dim),
            nn.ReLU(inplace=True),
        )
        # ER (corr=0.33): use low-level to avoid polluting high-level branch
        self.er_enc = nn.Sequential(
            nn.Linear(proj_dim, task_emb_dim),
            nn.ReLU(inplace=True),
        )

        # ── Auxiliary task heads ──────────────────────────────────────
        self.head_jsn = nn.Linear(task_emb_dim, num_jsn)
        self.head_op  = nn.Linear(task_emb_dim, num_op)
        self.head_er  = nn.Linear(task_emb_dim, num_er)

        # ── Cross-Attention Fusion ────────────────────────────────────
        # Q = v_kl = cat[v_low, v_high] (512d)
        # KV = [e_jsn, e_op, e_er] stacked as (B, 3, 64)
        self.cross_attn = LightCrossAttention(
            query_dim=proj_dim * 2,   # 512
            kv_dim=task_emb_dim,      # 64
        )

        # ── Main KL head ──────────────────────────────────────────────
        # Input: v_low(256) + v_high(256) + attn_ctx(64) = 576
        kl_in = proj_dim * 2 + task_emb_dim   # 576
        self.head_kl = nn.Sequential(
            nn.Linear(kl_in, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_kl),
        )

    def extract_features(self, x: torch.Tensor):
        f_low  = self.low_features(x)
        f_high = self.high_features(f_low)
        v_low  = self.pool(self.proj_low(f_low)).flatten(1)    # (B, 256)
        v_high = self.pool(self.proj_high(f_high)).flatten(1)  # (B, 256)
        return v_low, v_high

    def forward(self, x: torch.Tensor):
        v_low, v_high = self.extract_features(x)

        # Correlation-aware routing to task encoders
        e_jsn = self.jsn_enc(torch.cat([v_low, v_high], dim=1))  # (B, 64)
        e_op  = self.op_enc(v_high)                               # (B, 64)
        e_er  = self.er_enc(v_low)                                # (B, 64)

        # Auxiliary task predictions
        logit_jsn = self.head_jsn(e_jsn)
        logit_op  = self.head_op(e_op)
        logit_er  = self.head_er(e_er)

        # Cross-attention fusion:
        #   Q  = v_kl = cat[v_low, v_high]         (B, 512)
        #   KV = [e_jsn, e_op, e_er] as sequence   (B, 3, 64)
        v_kl     = torch.cat([v_low, v_high], dim=1)              # (B, 512)
        kv_stack = torch.stack([e_jsn, e_op, e_er], dim=1)        # (B, 3, 64)
        attn_ctx = self.cross_attn(v_kl, kv_stack)                # (B, 64)

        # KL head: MSFM features + attention-weighted auxiliary context
        kl_feat  = torch.cat([v_low, v_high, attn_ctx], dim=1)    # (B, 576)
        logit_kl = self.head_kl(kl_feat)                          # (B, num_kl)

        if not self.training:
            return logit_kl

        return {
            'kl':  logit_kl,
            'jsn': logit_jsn,
            'op':  logit_op,
            'er':  logit_er,
        }


class MobileNetV2MSFMMultitask(nn.Module):
    def __init__(self,
                 proj_dim: int = 256,
                 task_emb_dim: int = 64,
                 num_kl: int = 5,
                 num_jsn: int = 4,
                 num_op: int = 4,
                 num_er: int = 2):
        super().__init__()

        base = models.mobilenet_v2(weights='IMAGENET1K_V1')
        self.low_features  = base.features[:14]
        self.high_features = base.features[14:]
        self.pool = nn.AdaptiveAvgPool2d(1)

        self.proj_low  = Conv1x1(96,   proj_dim)
        self.proj_high = Conv1x1(1280, proj_dim)

        self.jsn_enc = nn.Sequential(
            nn.Linear(proj_dim * 2, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, task_emb_dim),
            nn.ReLU(inplace=True),
        )
        self.op_enc = nn.Sequential(
            nn.Linear(proj_dim, task_emb_dim),
            nn.ReLU(inplace=True),
        )
        self.er_enc = nn.Sequential(
            nn.Linear(proj_dim, task_emb_dim),
            nn.ReLU(inplace=True),
        )

        self.head_jsn = nn.Linear(task_emb_dim, num_jsn)
        self.head_op  = nn.Linear(task_emb_dim, num_op)
        self.head_er  = nn.Linear(task_emb_dim, num_er)

        self.w_jsn = nn.Parameter(torch.tensor(0.71))
        self.w_op  = nn.Parameter(torch.tensor(0.73))
        self.w_er  = nn.Parameter(torch.tensor(0.33))

        kl_in = proj_dim * 2 + task_emb_dim * 3   # 704
        self.head_kl = nn.Sequential(
            nn.Linear(kl_in, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_kl),
        )

    def extract_features(self, x: torch.Tensor):
        f_low  = self.low_features(x)
        f_high = self.high_features(f_low)
        v_low  = self.pool(self.proj_low(f_low)).flatten(1)
        v_high = self.pool(self.proj_high(f_high)).flatten(1)
        return v_low, v_high

    def forward(self, x: torch.Tensor):
        v_low, v_high = self.extract_features(x)

        e_jsn = self.jsn_enc(torch.cat([v_low, v_high], dim=1))
        e_op  = self.op_enc(v_high)
        e_er  = self.er_enc(v_low)

        logit_jsn = self.head_jsn(e_jsn)
        logit_op  = self.head_op(e_op)
        logit_er  = self.head_er(e_er)

        kl_feat = torch.cat([
            v_low, v_high,
            self.w_jsn * e_jsn,
            self.w_op  * e_op,
            self.w_er  * e_er,
        ], dim=1)                                                  # (B, 704)
        logit_kl = self.head_kl(kl_feat)

        if not self.training:
            return logit_kl

        return {
            'kl':  logit_kl,
            'jsn': logit_jsn,
            'op':  logit_op,
            'er':  logit_er,
        }

class MultitaskLoss(nn.Module):
    def __init__(self, asym_loss_fn,
                 w_jsn: float = 0.40,
                 w_op:  float = 0.41,
                 w_er:  float = 0.19):
        super().__init__()
        self.asym  = asym_loss_fn
        self.w_jsn = w_jsn
        self.w_op  = w_op
        self.w_er  = w_er

    def forward(self, outputs: dict, labels: dict):
        loss_kl  = self.asym(outputs['kl'], labels['kl'])
        loss_jsn = F.cross_entropy(outputs['jsn'], labels['jsn'].long(), ignore_index=-1)
        loss_op  = F.cross_entropy(outputs['op'],  labels['op'].long(),  ignore_index=-1)
        loss_er  = F.cross_entropy(outputs['er'],  labels['er'].long(),  ignore_index=-1)

        total = (loss_kl
                 + self.w_jsn * loss_jsn
                 + self.w_op  * loss_op
                 + self.w_er  * loss_er)

        detail = {
            'loss_kl':  loss_kl.item(),
            'loss_jsn': loss_jsn.item(),
            'loss_op':  loss_op.item(),
            'loss_er':  loss_er.item(),
        }
        return total, detail

class MobileNetV2MSFMMultitaskJSNOP(nn.Module):
    def __init__(self, proj_dim=256, task_emb_dim=64,
                 num_kl=5, num_jsn=4, num_op=4):
        super().__init__()

        base = models.mobilenet_v2(weights='IMAGENET1K_V1')
        self.low_features  = base.features[:14]
        self.high_features = base.features[14:]
        self.pool = nn.AdaptiveAvgPool2d(1)

        self.proj_low  = Conv1x1(96,   proj_dim)
        self.proj_high = Conv1x1(1280, proj_dim)

        self.jsn_enc = nn.Sequential(
            nn.Linear(proj_dim * 2, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, task_emb_dim),
            nn.ReLU(inplace=True),
        )

        self.op_enc = nn.Sequential(
            nn.Linear(proj_dim, task_emb_dim),
            nn.ReLU(inplace=True),
        )

        self.head_jsn = nn.Linear(task_emb_dim, num_jsn)
        self.head_op  = nn.Linear(task_emb_dim, num_op)

        self.w_jsn = nn.Parameter(torch.tensor(0.71))
        self.w_op  = nn.Parameter(torch.tensor(0.73))

        kl_in = proj_dim * 2 + task_emb_dim * 2   # 640
        self.head_kl = nn.Sequential(
            nn.Linear(kl_in, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_kl),
        )

    def extract_features(self, x: torch.Tensor):
        f_low  = self.low_features(x)
        f_high = self.high_features(f_low)
        v_low  = self.pool(self.proj_low(f_low)).flatten(1)
        v_high = self.pool(self.proj_high(f_high)).flatten(1)
        return v_low, v_high

    def forward(self, x: torch.Tensor):
        v_low, v_high = self.extract_features(x)

        e_jsn = self.jsn_enc(torch.cat([v_low, v_high], dim=1))
        e_op  = self.op_enc(v_high)

        logit_jsn = self.head_jsn(e_jsn)
        logit_op  = self.head_op(e_op)

        kl_feat = torch.cat([
            v_low, v_high,
            self.w_jsn * e_jsn,
            self.w_op  * e_op,
        ], dim=1)                                                  # (B, 640)
        logit_kl = self.head_kl(kl_feat)

        if not self.training:
            return logit_kl

        return {
            'kl':  logit_kl,
            'jsn': logit_jsn,
            'op':  logit_op,
        }


class MultitaskLossJSNOP(nn.Module):
    def __init__(self, asym_loss_fn,
                 w_jsn: float = 0.49,
                 w_op:  float = 0.51):
        super().__init__()
        self.asym  = asym_loss_fn
        self.w_jsn = w_jsn
        self.w_op  = w_op

    def forward(self, outputs: dict, labels: dict):
        loss_kl  = self.asym(outputs['kl'], labels['kl'])
        loss_jsn = F.cross_entropy(outputs['jsn'], labels['jsn'].long(),
                                   ignore_index=-1)
        loss_op  = F.cross_entropy(outputs['op'],  labels['op'].long(),
                                   ignore_index=-1)

        total = loss_kl + self.w_jsn * loss_jsn + self.w_op * loss_op

        detail = {
            'loss_kl':  loss_kl.item(),
            'loss_jsn': loss_jsn.item(),
            'loss_op':  loss_op.item(),
        }
        return total, detail

class MobileNetV2MSFMMultitaskOP(nn.Module):
    def __init__(self, proj_dim=256, task_emb_dim=64,
                 num_kl=5, num_op=4):
        super().__init__()

        base = models.mobilenet_v2(weights='IMAGENET1K_V1')
        self.low_features  = base.features[:14]
        self.high_features = base.features[14:]
        self.pool = nn.AdaptiveAvgPool2d(1)

        self.proj_low  = Conv1x1(96,   proj_dim)
        self.proj_high = Conv1x1(1280, proj_dim)

        self.op_enc = nn.Sequential(
            nn.Linear(proj_dim, task_emb_dim),
            nn.ReLU(inplace=True),
        )
        self.head_op = nn.Linear(task_emb_dim, num_op)

        self.w_op = nn.Parameter(torch.tensor(0.73))

        kl_in = proj_dim * 2 + task_emb_dim   # 576
        self.head_kl = nn.Sequential(
            nn.Linear(kl_in, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_kl),
        )

    def extract_features(self, x):
        f_low  = self.low_features(x)
        f_high = self.high_features(f_low)
        v_low  = self.pool(self.proj_low(f_low)).flatten(1)
        v_high = self.pool(self.proj_high(f_high)).flatten(1)
        return v_low, v_high

    def forward(self, x):
        v_low, v_high = self.extract_features(x)
        e_op = self.op_enc(v_high)
        logit_op = self.head_op(e_op)

        kl_feat  = torch.cat([v_low, v_high, self.w_op * e_op], dim=1)
        logit_kl = self.head_kl(kl_feat)

        if not self.training:
            return logit_kl

        return {'kl': logit_kl, 'op': logit_op}


class MultitaskLossOP(nn.Module):
    def __init__(self, asym_loss_fn, w_op: float = 1.0):
        super().__init__()
        self.asym = asym_loss_fn
        self.w_op = w_op

    def forward(self, outputs, labels):
        loss_kl = self.asym(outputs['kl'], labels['kl'])
        loss_op = F.cross_entropy(outputs['op'], labels['op'].long(),
                                  ignore_index=-1)
        total = loss_kl + self.w_op * loss_op
        detail = {'loss_kl': loss_kl.item(), 'loss_op': loss_op.item()}
        return total, detail

class MobileNetV2MSFMMultitaskJSN(nn.Module):
    def __init__(self, proj_dim=256, task_emb_dim=64,
                 num_kl=5, num_jsn=4):
        super().__init__()

        base = models.mobilenet_v2(weights='IMAGENET1K_V1')
        self.low_features  = base.features[:14]
        self.high_features = base.features[14:]
        self.pool = nn.AdaptiveAvgPool2d(1)

        self.proj_low  = Conv1x1(96,   proj_dim)
        self.proj_high = Conv1x1(1280, proj_dim)

        self.jsn_enc = nn.Sequential(
            nn.Linear(proj_dim * 2, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, task_emb_dim),
            nn.ReLU(inplace=True),
        )
        self.head_jsn = nn.Linear(task_emb_dim, num_jsn)

        self.w_jsn = nn.Parameter(torch.tensor(0.71))

        kl_in = proj_dim * 2 + task_emb_dim   # 576
        self.head_kl = nn.Sequential(
            nn.Linear(kl_in, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_kl),
        )

    def extract_features(self, x):
        f_low  = self.low_features(x)
        f_high = self.high_features(f_low)
        v_low  = self.pool(self.proj_low(f_low)).flatten(1)
        v_high = self.pool(self.proj_high(f_high)).flatten(1)
        return v_low, v_high

    def forward(self, x):
        v_low, v_high = self.extract_features(x)
        e_jsn = self.jsn_enc(torch.cat([v_low, v_high], dim=1))
        logit_jsn = self.head_jsn(e_jsn)

        kl_feat  = torch.cat([v_low, v_high, self.w_jsn * e_jsn], dim=1)
        logit_kl = self.head_kl(kl_feat)

        if not self.training:
            return logit_kl

        return {'kl': logit_kl, 'jsn': logit_jsn}


class MultitaskLossJSN(nn.Module):
    def __init__(self, asym_loss_fn, w_jsn: float = 1.0):
        super().__init__()
        self.asym  = asym_loss_fn
        self.w_jsn = w_jsn

    def forward(self, outputs, labels):
        loss_kl  = self.asym(outputs['kl'], labels['kl'])
        loss_jsn = F.cross_entropy(outputs['jsn'], labels['jsn'].long(),
                                   ignore_index=-1)
        total = loss_kl + self.w_jsn * loss_jsn
        detail = {'loss_kl': loss_kl.item(), 'loss_jsn': loss_jsn.item()}
        return total, detail

def get_model(model_name: str, num_classes: int = 5, **kwargs) -> nn.Module:

    if model_name == 'resnet50':
        model = models.resnet50(weights='IMAGENET1K_V1')
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features, num_classes)
        )

    elif model_name == 'resnet101':
        model = models.resnet101(weights='IMAGENET1K_V1')
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features, num_classes)
        )

    elif model_name == 'densenet201':
        model = models.densenet201(weights='IMAGENET1K_V1')
        in_features = model.classifier.in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features, num_classes)
        )

    elif model_name == 'efficientnet_b4':
        model = models.efficientnet_b4(weights='IMAGENET1K_V1')
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features, num_classes)
        )

    elif model_name == 'mobilenet_v2':
        model = models.mobilenet_v2(weights='IMAGENET1K_V1')
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features, num_classes)
        )

    elif model_name == 'mobilenet_v3':
        model = models.mobilenet_v3_large(weights='IMAGENET1K_V1')
        in_features = model.classifier[3].in_features
        model.classifier[3] = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features, num_classes)
        )

    elif model_name == 'mobilenet_v2_CBAM':
        model = MobileNetV2WithCBAM(num_classes=num_classes)

    elif model_name == 'mobilenet_v2_MSFM':
        model = MobileNetV2WithMSFM(num_classes=num_classes)

    elif model_name == 'mobilenet_v2_ORM':
        backbone = models.mobilenet_v2(weights='IMAGENET1K_V1')
        in_features = backbone.classifier[1].in_features
        backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features, 1),
        )
        orm   = OrdinalRegressionModule(num_classes=num_classes)
        model = _ORMBackbone(backbone, orm)

    elif model_name == 'mobilenet_v2_CORAL':
        backbone = models.mobilenet_v2(weights='IMAGENET1K_V1')
        in_features = backbone.classifier[1].in_features
        backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features, 1),
        )
        coral = CORALLayer(num_classes=num_classes)
        model = _CORALBackbone(backbone, coral)

    elif model_name == 'mobilenet_v2_CORAL_wide':
        model = MobileNetV2CORALWide(num_classes=num_classes)
    
    elif model_name == 'mobilenet_v2_CORN':
        backbone = models.mobilenet_v2(weights='IMAGENET1K_V1')
        in_features = backbone.classifier[1].in_features
        backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features, num_classes - 1),
        )
        model = backbone

    elif model_name == 'mobilenet_v2_MSFM_CORN':
        model = MobileNetV2WithMSFMCORN(num_classes=num_classes)
    
    elif model_name == 'mobilenet_v2_MSFM_ordinal':
        model = MobileNetV2WithMSFMOrdinal(num_classes=num_classes)

    elif model_name == 'mobilenet_v2_MSFM_multitask':
        model = MobileNetV2MSFMMultitask(
            num_kl=num_classes,
            num_jsn=kwargs.get('num_jsn', 4),
            num_op=kwargs.get('num_op', 4),
            num_er=kwargs.get('num_er', 2),
            proj_dim=kwargs.get('proj_dim', 256),
            task_emb_dim=kwargs.get('task_emb_dim', 64),
        )

    elif model_name == 'mobilenet_v2_MSFM_multitask_jsn_op':
        model = MobileNetV2MSFMMultitaskJSNOP(
            num_kl=num_classes,
            num_jsn=kwargs.get('num_jsn', 4),
            num_op=kwargs.get('num_op', 4),
            proj_dim=kwargs.get('proj_dim', 256),
            task_emb_dim=kwargs.get('task_emb_dim', 64),
        )

    elif model_name == 'mobilenet_v2_MSFM_multitask_op':
        model = MobileNetV2MSFMMultitaskOP(
            num_kl=num_classes,
            num_op=kwargs.get('num_op', 4),
            proj_dim=kwargs.get('proj_dim', 256),
            task_emb_dim=kwargs.get('task_emb_dim', 64),
        )

    elif model_name == 'mobilenet_v2_MSFM_multitask_jsn':
        model = MobileNetV2MSFMMultitaskJSN(
            num_kl=num_classes,
            num_jsn=kwargs.get('num_jsn', 4),
            proj_dim=kwargs.get('proj_dim', 256),
            task_emb_dim=kwargs.get('task_emb_dim', 64),
        )

    elif model_name == 'mobilenet_v2_MSFM_multitask_crossattn':
        model = MobileNetV2MSFMMultitaskCrossAttn(
            num_kl=num_classes,
            num_jsn=kwargs.get('num_jsn', 4),
            num_op=kwargs.get('num_op', 4),
            num_er=kwargs.get('num_er', 2),
            proj_dim=kwargs.get('proj_dim', 256),
            task_emb_dim=kwargs.get('task_emb_dim', 64),
        )

    else:
        raise ValueError(
            f"Unknown model: '{model_name}'. "
            f"Choose from: resnet50, resnet101, densenet201, efficientnet_b4, "
            f"mobilenet_v2, mobilenet_v3, mobilenet_v2_CBAM, mobilenet_v2_MSFM, "
            f"mobilenet_v2_ORM, mobilenet_v2_CORAL, mobilenet_v2_CORAL_wide, "
            f"mobilenet_v2_CORN, mobilenet_v2_MSFM_CORN, mobilenet_v2_MSFM_ordinal, "
            f"mobilenet_v2_MSFM_multitask, mobilenet_v2_MSFM_multitask_jsn_op, "
            f"mobilenet_v2_MSFM_multitask_op, mobilenet_v2_MSFM_multitask_jsn, "
            f"mobilenet_v2_MSFM_multitask_crossattn"
        )

    return model