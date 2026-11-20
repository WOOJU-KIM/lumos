"""Original research TCN architecture, CPU-only and deterministic."""
from common import add_torch_dependency
add_torch_dependency()
import torch
from torch import nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, bias):
        super().__init__()
        self.proj = nn.Conv1d(72, 16, 1)
        self.layers = nn.ModuleList([nn.Conv1d(16, 16, 2, dilation=d) for d in [1, 2, 4, 8, 16]])
        self.dropout = nn.Dropout(.05)
        self.head = nn.Linear(16, 2)
        with torch.no_grad():
            self.head.bias.copy_(torch.tensor(bias, dtype=torch.float32))

    def forward(self, x):
        x = F.relu(self.proj(x))
        for d, layer in zip([1, 2, 4, 8, 16], self.layers):
            x = F.relu(x + self.dropout(layer(F.pad(x, (d, 0)))))
        return self.head(x[:, :, -1])
