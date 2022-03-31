import torch
import torch.nn as nn
import torch.nn.functional as F
from src.fspool import FSPool
############
# Encoders #
############
"""
    Clase FSEncoder, con la operacion invariante de fspool (Feature wise sort pool)
"""
class FSEncoderDSPN(nn.Module):
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
"""
    Clase FSEncoder, con la operacion invariante de fspool (Feature wise sort pool), pero adaptado
    a la clasificacións
"""
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
"""
    Clase SumEncoder, con la operacion invariante suma
"""
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

"""
    Clase SumEncoder, con la operacion invariante suma, adaptado para clasificar
"""
class SumEncoderDSPN(nn.Module):
    def __init__(self, input_channels, output_channels, dim, **kwargs):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(input_channels + 1, dim, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(dim, dim, 1),
        )

    def forward(self, x, mask, *args):
        mask = mask.unsqueeze(1)
        x = torch.cat([x, mask], dim=1)  # include mask as part of set
        x = self.conv(x)
        x = x.sum(2)
        return x
"""
    Clase MaxEncoder, con la operacion invariante Max
"""
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
"""
    Clase MaxEncoder, con la operacion invariante Max, adaptado a la clasificación
"""
class MaxEncoderDSPN(nn.Module):
    def __init__(self, input_channels, output_channels, dim, **kwargs):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(input_channels+ 1, dim, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(dim, dim, 1),
        )

    def forward(self, x, mask, *args):
        mask = mask.unsqueeze(1)
        x = torch.cat([x, mask], dim=1)  # include mask as part of set
        x = self.conv(x)
        x = x.max(2)[0]
        return x
"""
    Clase MeanEncoderClasification, con la operacion invariante Mean, adaptado a la clasificación
"""
class MeanEncoder(nn.Module):
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
        x = x.sum(2) / n_points.size(1)
        x = self.lin(x)
        x = self.classifier(x)
        return x
"""
    Clase MeanEncoder, con la operacion invariante Mean
"""
class MeanEncoderDSPN(nn.Module):
    def __init__(self, input_channels, output_channels, dim, **kwargs):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(input_channels + 1, dim, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(dim, dim, 1),
        )

    def forward(self, x, n_points, *args):
        mask = mask.unsqueeze(1)
        x = torch.cat([x, mask], dim=1)  # include mask as part of set
        x = self.conv(x)
        x = x.sum(2) / n_points.size(1)
        return x