import torch
import torch.nn as nn
from torch.nn import Module, Conv2d, Parameter, Softmax
from sync_batchnorm import SynchronizedBatchNorm2d as BatchNorm2d


class PAM_Module(Module):
    """ Position attention module"""
    def __init__(self, in_dim):
        super(PAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = Parameter(torch.zeros(1))

        self.softmax = Softmax(dim=-1)
    def forward(self, x):
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width*height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width*height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)

        proj_value = self.value_conv(x).view(m_batchsize, -1, width*height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma*out + x
        return out


class attentionHead(nn.Module):
    def __init__(self, in_channels):
        super(attentionHead, self).__init__()
        self.conv5a = nn.Sequential(nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
                                    BatchNorm2d(in_channels),
                                    nn.ReLU())
        self.sa = PAM_Module(in_channels)


    def forward(self, x):
        feat1 = self.conv5a(x)
        sa_feat = self.sa(feat1)
        sa_output = self.conv51(sa_feat)

        return sa_output

