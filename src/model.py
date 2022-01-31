import torch
import torch.nn as nn
import torch.nn.functional as F

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
        self.salida = nn.Linear(64, 10)

    def forward(self, x, mask=None):
        mask = mask.unsqueeze(1)
        x = torch.cat([x, mask], dim=1)  # include mask as part of set
        x = self.conv(x)
        x = x / x.size(2)  # normalise so that activations aren't too high with big sets
        x, _ = self.pool(x)
        x = self.salida(x)
        x = F.log_softmax(x)
        return x