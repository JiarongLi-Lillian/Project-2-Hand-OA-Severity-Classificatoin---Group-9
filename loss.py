import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import Counter


def get_class_weights(labels, num_classes=5, device='cpu'):
    class_counts = Counter(labels)
    total = len(labels)
    weights = [total / (num_classes * max(class_counts[i], 1))
               for i in range(num_classes)]
    return torch.tensor(weights, dtype=torch.float32).to(device)


def build_asymmetric_loss_matrix(num_classes=5, alpha_L=2.0, alpha_R=0.5):
    W = torch.zeros(num_classes, num_classes)
    for j in range(num_classes):
        for k in range(num_classes):
            d = j - k
            if d > 0:           # underestimation: penalize more heavily
                W[j, k] = d + alpha_L * d ** 2
            elif d < 0:         # overestimation: penalize less heavily
                d = abs(d)
                W[j, k] = d + alpha_R * d ** 2
    W = W / W.max()
    return W


class OrdinalAsymmetricLoss(nn.Module):
    def __init__(self, num_classes=5, alpha_L=2.0, alpha_R=0.5,
                 class_weights=None, already_probs=False):
        super().__init__()
        self.num_classes   = num_classes
        self.alpha_L       = alpha_L
        self.alpha_R       = alpha_R
        self.already_probs = already_probs

        W = build_asymmetric_loss_matrix(num_classes, alpha_L, alpha_R)
        self.register_buffer('W', W)

        if class_weights is not None:
            self.register_buffer('class_weights', class_weights.float())
        else:
            self.class_weights = None

    def forward(self, inputs, targets):
        if self.already_probs:
            probs = inputs
        else:
            probs = F.softmax(inputs, dim=1)

        W_true          = self.W[targets]
        loss_per_sample = (W_true * probs).sum(dim=1)

        if self.class_weights is not None:
            loss_per_sample = loss_per_sample * self.class_weights[targets]

        return loss_per_sample.mean()

    def extra_repr(self):
        return (f'num_classes={self.num_classes}, '
                f'alpha_L={self.alpha_L}, alpha_R={self.alpha_R}, '
                f'class_weights={self.class_weights is not None}')


class CORALLoss(nn.Module):
    def __init__(self, num_classes=5, class_weights=None, task_weights=None):
        super().__init__()
        self.num_classes = num_classes

        if task_weights is not None:
            tw = torch.tensor(task_weights, dtype=torch.float32)
            assert len(tw) == num_classes - 1, \
                f"task_weights must have length num_classes-1={num_classes-1}"
            assert (tw > 0).all(), \
                "all task_weights must be > 0 to guarantee rank consistency"
        else:
            tw = torch.ones(num_classes - 1, dtype=torch.float32)
        self.register_buffer('task_weights', tw)

        if class_weights is not None:
            self.register_buffer('class_weights', class_weights.float())
        else:
            self.class_weights = None

    def forward(self, logits, targets):
        K = self.num_classes

        # Build binary label matrix: binary_targets[i, k] = 1 if targets[i] > k
        ranks = torch.arange(K - 1, device=logits.device).unsqueeze(0)  # (1, K-1)
        binary_targets = (targets.unsqueeze(1) > ranks).float()          # (B, K-1)

        # Binary cross-entropy per task, numerically stable
        loss_per_task = F.binary_cross_entropy_with_logits(
            logits, binary_targets, reduction='none'
        )                                                                 # (B, K-1)

        # Apply non-uniform task weights to amplify minority-class tasks
        loss_per_task = loss_per_task * self.task_weights.unsqueeze(0)   # (B, K-1)

        # Average across tasks
        loss_per_sample = loss_per_task.mean(dim=1)                      # (B,)

        # Optional per-sample weighting by class frequency
        if self.class_weights is not None:
            loss_per_sample = loss_per_sample * self.class_weights[targets]

        return loss_per_sample.mean()

    def extra_repr(self):
        return (f'num_classes={self.num_classes}, '
                f'task_weights={self.task_weights.tolist()}, '
                f'class_weights={self.class_weights is not None}')


class CORNLoss(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, logits, targets):
        K = self.num_classes
        loss = torch.zeros(1, device=logits.device)
        num_examples = 0

        for task_index in range(K - 1):
            mask = targets >= task_index
            if mask.sum() < 1:
                continue
            logits_task  = logits[mask, task_index]
            targets_task = (targets[mask] > task_index).float()
            loss += F.binary_cross_entropy_with_logits(
                logits_task, targets_task, reduction='sum'
            )
            num_examples += mask.sum().item()

        return loss / num_examples if num_examples > 0 else loss

class RankConsistencyLoss(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, logits, targets):
        probs     = F.softmax(logits, dim=1)           # (B, K)
        cum_probs = torch.cumsum(probs, dim=1)         # (B, K), P(y<=k)

        # Binary ordinal targets: target<=k → 1, target>k → 0
        K     = self.num_classes
        ranks = torch.arange(K, device=logits.device).unsqueeze(0)  # (1, K)
        binary_targets = (targets.unsqueeze(1) <= ranks).float()     # (B, K)

        return F.binary_cross_entropy(
            cum_probs.clamp(1e-6, 1 - 1e-6), binary_targets
        )

class AsymOrdinalLoss(nn.Module):
    def __init__(self, num_classes=5, alpha_L=2.0, alpha_R=0.5,
                 class_weights=None, lam=0.3):
        super().__init__()
        self.lam  = lam
        self.asym = OrdinalAsymmetricLoss(
            num_classes=num_classes,
            alpha_L=alpha_L,
            alpha_R=alpha_R,
            class_weights=class_weights,
            already_probs=False,
        )
        self.rank = RankConsistencyLoss(num_classes=num_classes)

    def forward(self, logits, targets):
        return self.asym(logits, targets) + self.lam * self.rank(logits, targets)

    def extra_repr(self):
        return f'lam={self.lam}'

def omega_ord_map_predict(probs, W):
    """Predict class by minimizing expected asymmetric loss."""
    expected_losses = probs @ W.T
    return expected_losses.argmin(dim=1)


def get_loss(loss_name, labels=None, device='cpu', num_classes=5,
             already_probs=False, **kwargs):
    if loss_name == 'crossentropy':
        if labels is not None:
            weights = get_class_weights(labels, num_classes, device)
            return nn.CrossEntropyLoss(weight=weights)
        return nn.CrossEntropyLoss()

    elif loss_name == 'ordinal_asymmetric':
        alpha_L       = kwargs.get('alpha_L', 2.0)
        alpha_R       = kwargs.get('alpha_R', 0.5)
        class_weights = get_class_weights(labels, num_classes, device) if labels is not None else None
        return OrdinalAsymmetricLoss(
            num_classes=num_classes,
            alpha_L=alpha_L,
            alpha_R=alpha_R,
            class_weights=class_weights,
            already_probs=already_probs
        ).to(device)

    elif loss_name == 'coral':
        task_weights  = kwargs.get('task_weights', None)
        class_weights = get_class_weights(labels, num_classes, device) if labels is not None else None
        return CORALLoss(
            num_classes=num_classes,
            class_weights=class_weights,
            task_weights=task_weights,
        ).to(device)
    
    elif loss_name == 'corn':
        return CORNLoss(num_classes=num_classes).to(device)

    elif loss_name == 'asym_ordinal':
        alpha_L       = kwargs.get('alpha_L', 2.0)
        alpha_R       = kwargs.get('alpha_R', 0.5)
        lam           = kwargs.get('lam', 0.1)
        class_weights = get_class_weights(labels, num_classes, device) if labels is not None else None
        return AsymOrdinalLoss(
            num_classes=num_classes,
            alpha_L=alpha_L,
            alpha_R=alpha_R,
            class_weights=class_weights,
            lam=lam,
        ).to(device)

    else:
        raise ValueError(
            f"Unknown loss: '{loss_name}'. "
            f"Choose from: 'crossentropy', 'ordinal_asymmetric', 'coral', 'corn', 'asym_ordinal'"
        )