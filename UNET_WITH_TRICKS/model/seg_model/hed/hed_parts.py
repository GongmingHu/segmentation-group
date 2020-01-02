import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class make_bilinear_weights(nn.Module):
    def __init__(self, size, num_channels):
        super(make_bilinear_weights, self).__init__()
        self.size = size
        self.num_channels = num_channels

    def forward(self, *input):
        factor = (self.size + 1) // 2
        if self.size % 2 == 1:
            center = factor - 1
        else:
            center = factor - 0.5
        og = np.ogrid[:self.size, :self.size]
        filt = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)
        filt = torch.from_numpy(filt)
        w = torch.zeros(self.num_channels, self.num_channels, self.size, self.size)
        w.requires_grad = False  # Set not trainable.
        for i in range(self.num_channels):
            for j in range(self.num_channels):
                if i == j:
                    w[i, j] = filt
        return w