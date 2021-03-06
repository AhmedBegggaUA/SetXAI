import torch
import torch.nn.functional as F
import scipy.optimize
import seaborn as sn
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
def per_sample_hungarian_loss(sample_np):
    row_idx, col_idx = scipy.optimize.linear_sum_assignment(sample_np)
    return row_idx, col_idx
def hungarian_loss(predictions, targets):
        # predictions and targets shape :: (n, c, s)
        predictions, targets = outer(predictions, targets)
        # squared_error shape :: (n, s, s)
        squared_error = (predictions - targets).pow(2).mean(1)
        squared_error_np = squared_error.detach().cpu().numpy()
        indices = map(per_sample_hungarian_loss, squared_error_np)
        losses = [sample[row_idx, col_idx].mean() for sample, (row_idx, col_idx) in zip(squared_error, indices)]
        total_loss = torch.mean(torch.stack(list(losses)))
        return total_loss

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
def get_weight(modelo):
    fig, ax = plt.subplots(2, 3, figsize=(20,10))
    j = 1
    for i ,w in enumerate(modelo.named_parameters()):
        if i % 2 == 0 and i <= 10: 
            plt.subplot(2,3,j) 
            plt.title(w[0])
            if w[1].size(-1) == 1:
                data = w[1].squeeze(2).detach().numpy()
            sn.heatmap(data, annot=False, fmt='g')
            j+=1
            if i == 10:
                break
    plt.show()