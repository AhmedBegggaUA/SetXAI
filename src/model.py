import torch
import torch.nn as nn
import torch.nn.functional as F
from fspool import FSPool
############
# Encoders #
############
class FSEncoder(nn.Module):
    def __init__(self, input_channels, output_channels, dim):
        super().__init__()
        for m in self.modules():
            if (
                isinstance(m, nn.Linear)
                or isinstance(m, nn.Conv2d)
                or isinstance(m, nn.Conv1d)
            ):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
        self.conv = nn.Sequential(
            nn.Conv1d(input_channels + 1, dim, 1),
            nn.ReLU(),
            nn.Conv1d(dim, dim, 1),
            nn.ReLU(),
            nn.Conv1d(dim, output_channels, 1),
        )
        
        self.pool = FSPool(output_channels, 20, relaxed=False)

    def forward(self, x, mask=None):
        mask = mask.unsqueeze(1)
        x = torch.cat([x, mask], dim=1)  # include mask as part of set
        x = self.conv(x)
        x = x / x.size(2)  # normalise so that activations aren't too high with big sets
        x, _ = self.pool(x)
        return x

class FSEncoderClasification(nn.Module):
    def __init__(self, input_channels, output_channels, dim):
        super().__init__()
        for m in self.modules():
            if (
                isinstance(m, nn.Linear)
                or isinstance(m, nn.Conv2d)
                or isinstance(m, nn.Conv1d)
            ):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
        self.conv = nn.Sequential(
            nn.Conv1d(input_channels, dim, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(dim, dim, 1),
        )
        self.lin = nn.Sequential(
            nn.Linear(dim, dim, 1),
            nn.ReLU(inplace=True),
            nn.Linear(dim, output_channels, 1),
        )
        self.classifier = nn.Sequential(
            nn.Linear(output_channels, output_channels),
            nn.ReLU(),
            nn.Linear(output_channels, 10),
        )
        self.pool = FSPool(dim, 20, relaxed=True)

    def forward(self, x, mask=None):
        x = self.conv(x)
        x, perm = self.pool(x)
        x = self.lin(x)
        x = self.classifier(x)
        return x

class SumEncoder(nn.Module):
    def __init__(self, input_channels, output_channels, dim, **kwargs):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(input_channels, dim, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(dim, dim, 1),
        )
        self.lin = nn.Sequential(
            nn.Linear(dim, dim, 1),
            nn.ReLU(inplace=True),
            nn.Linear(dim, output_channels, 1),
        )
        self.classifier = nn.Sequential(
            nn.Linear(output_channels, output_channels),
            nn.ReLU(),
            nn.Linear(output_channels, 10),
        )

    def forward(self, x, n_points, *args):
        x = self.conv(x)
        x = x.sum(2)
        x = self.lin(x)
        x = self.classifier(x)
        return x

class MaxEncoder(nn.Module):
    def __init__(self, input_channels, output_channels, dim, **kwargs):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(input_channels, dim, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(dim, dim, 1),
        )
        self.lin = nn.Sequential(
            nn.Linear(dim, dim, 1),
            nn.ReLU(inplace=True),
            nn.Linear(dim, output_channels, 1),
        )
        self.classifier = nn.Sequential(
            nn.Linear(output_channels, output_channels),
            nn.ReLU(),
            nn.Linear(output_channels, 10),
        )

    def forward(self, x, n_points, *args):
        x = self.conv(x)
        x = x.max(2)[0]
        x = self.lin(x)
        x = self.classifier(x)
        return x

class MeanEncoder(nn.Module):
    def __init__(self, *, input_channels, output_channels, dim, **kwargs):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(input_channels, dim, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(dim, dim, 1),
        )
        self.lin = nn.Sequential(
            nn.Linear(dim, dim, 1),
            nn.ReLU(inplace=True),
            nn.Linear(dim, output_channels, 1),
        )
        self.classifier = nn.Sequential(
            nn.Linear(output_channels, output_channels),
            nn.ReLU(),
            nn.Linear(output_channels, 10),
        )

    def forward(self, x, n_points, *args):
        x = self.conv(x)
        x = x.sum(2) / n_points.unsqueeze(1).float()
        x = self.lin(x)
        x = self.classifier(x)
        return x