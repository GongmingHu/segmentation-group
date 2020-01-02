import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import numpy as np
import cv2
import matplotlib.pyplot as plt


class dilatedweightBCE(nn.Module):
    def __init__(self, kernel_size=3, bg_weight=1, dilated_bg_weight=5, edge_weight=20):
        super(dilatedweightBCE, self).__init__()
        self.bg_weight = bg_weight
        self.dilated_bg_weight = dilated_bg_weight
        self.edge_weight = edge_weight

        self.kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))

    def forward(self, pred, target, **kwargs):
        # 每个类别的weight
        b, c, h, w = target.shape
        # num_pos = torch.sum(target, dim=[0, 1, 2, 3]).float()  # Shape: [b,]
        # num_neg = b * c * h * w - num_pos  # Shape: [b,]
        # pos_weight = num_neg / (num_pos + num_neg)
        # neg_weight = num_pos / (num_pos + num_neg)

        # 膨胀过的gt
        numpy_target = target.detach().cpu().numpy()
        dilated_target = np.zeros_like(numpy_target)
        for i in range(b):
            single_image = numpy_target[i, 0, :, :]
            dilated_image = cv2.dilate(single_image, self.kernel)
            dilated_target[i, 0, :, :] = dilated_image
        dilated_target = torch.FloatTensor(dilated_target).cuda()

        # 指定权重
        weight = torch.zeros_like(target)
        weight[dilated_target > 0.5] = self.dilated_bg_weight
        weight[dilated_target <= 0.5] = self.bg_weight
        weight[target > 0.5] = self.edge_weight

        # Calculate loss
        losses = F.binary_cross_entropy_with_logits(pred, target, weight=weight, reduction='none')
        # plt.imshow(losses.detach().cpu().numpy()[0, 0, :, :])
        # plt.show()
        loss = torch.mean(losses)
        return loss


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=True, weight=20, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce
        self.weight = weight

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, pos_weight=torch.tensor(self.weight), reduction='none')
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, pos_weight=torch.tensor(self.weight), reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, output, target):
        prediction = torch.sigmoid(output)
        intersection = torch.sum(prediction * target)
        union = torch.sum(prediction) + torch.sum(target) + 1e-7
        return 1 - 2 * intersection / union


class FocalDiceLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=True, weight=20, reduce=True):
        super(FocalDiceLoss, self).__init__()
        self.focal = FocalLoss(alpha, gamma, logits, weight, reduce)
        self.dice = DiceLoss()

    def forward(self, pred, target):
        dice = self.dice(pred, target)
        focal = self.focal(pred, target)
        return dice + focal
