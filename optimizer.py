import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR


def get_optimizer(optimizer_name, model, lr=1e-4, weight_decay=1e-5, orm_lr=None):
    orm_params      = [p for n, p in model.named_parameters()
                       if 'cut_points' in n and p.requires_grad]
    backbone_params = [p for n, p in model.named_parameters()
                       if 'cut_points' not in n and p.requires_grad]

    if orm_lr is not None and len(orm_params) > 0:
        param_groups = [
            {'params': backbone_params, 'lr': lr},
            {'params': orm_params,      'lr': orm_lr},
        ]
    else:
        param_groups = [{'params': backbone_params, 'lr': lr}]

    if optimizer_name == 'adam':
        return optim.Adam(param_groups, weight_decay=weight_decay)

    elif optimizer_name == 'sgd':
        return optim.SGD(param_groups, momentum=0.9, weight_decay=weight_decay)

    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")


def get_scheduler(scheduler_name, optimizer):

    if scheduler_name == 'plateau':
        return ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5,
        )

    elif scheduler_name == 'cosine':
        return CosineAnnealingLR(
            optimizer,
            T_max=50,
            eta_min=1e-6
        )

    else:
        raise ValueError(f"Unknown scheduler: {scheduler_name}")
        