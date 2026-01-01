import numpy as np
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, bandwidth):
        super().__init__()
        self.bandwidth = bandwidth
        self.frequencies = 2.0 ** torch.linspace(0.0, bandwidth - 1, bandwidth) * np.pi

        # self.register_buffer('frequencies', freqs)

    def forward(self, x):
        out = [x]
        for frequency in self.frequencies:
            out.append(torch.sin(x * frequency).type(torch.float32))
            out.append(torch.cos(x * frequency).type(torch.float32))
            
        return torch.cat(out, dim=-1)