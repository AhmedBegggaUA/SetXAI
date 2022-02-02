import torch
import torch.nn.functional as F

def chamfer_loss(predictions, targets):
    # predictions and targets shape :: (k, n, c, s)
    predictions, targets = outer(predictions, targets)
    # squared_error shape :: (k, n, s, s)
    squared_error = F.smooth_l1_loss(predictions, targets.expand_as(predictions), reduction="none").mean(2)
    loss = squared_error.min(2)[0] + squared_error.min(3)[0]
    return loss.view(loss.size(0), -1).mean(1)

def outer(a, b=None):
    """ Compute outer product between a and b (or a and a if b is not specified). """
    if b is None:
        b = a
    size_a = tuple(a.size()) + (b.size()[-1],)
    size_b = tuple(b.size()) + (a.size()[-1],)
    a = a.unsqueeze(dim=-1).expand(*size_a)
    b = b.unsqueeze(dim=-2).expand(*size_b)
    return a, b