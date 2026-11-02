"""The same input-row-normalized objective for every arm and DDP microbatch."""
import torch
import torch.nn.functional as F


def weighted_ce_per_row(logits, target, weights=None):
    if target.ndim != 1 or logits.shape[0] != target.shape[0] or not target.numel():
        raise ValueError('Require one scalar target per nonempty input row')
    weight = None if weights is None else torch.as_tensor(weights, dtype=torch.float32, device=logits.device)
    return F.cross_entropy(logits.float(), target, weight=weight,
                           ignore_index=-100, reduction='sum') / target.shape[0]
