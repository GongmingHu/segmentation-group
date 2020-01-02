import torch
import torch.nn as nn
from torch.nn import Parameter
from torch.nn import functional as F
from sync_batchnorm import SynchronizedBatchNorm2d as BatchNorm2d


class PPM(nn.Module):
    def __init__(self, in_dim, reduction_dim, bins):
        super(PPM, self).__init__()
        self.features = []
        for bin in bins:
            self.features.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(bin),
                nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
                BatchNorm2d(reduction_dim),
                nn.ReLU(inplace=True)
            ))
        self.features = nn.ModuleList(self.features)
        self.gamma = Parameter(torch.zeros(1))
        self.conv51 = nn.Sequential(nn.Conv2d(512, 512, 3, padding=1, bias=False),
                                    BatchNorm2d(512),
                                    nn.ReLU())

    def forward(self, x):
        x_size = x.size()
        out = []
        for f in self.features:
            out.append(F.interpolate(f(x), x_size[2:], mode='bilinear', align_corners=True))
        out = self.gamma*(torch.cat(out, 1)) + x
        out = self.conv51(out)
        return out