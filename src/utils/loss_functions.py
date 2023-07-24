import torch
from torch import nn


class WGLoss(nn.Module):
    def __init__(self, white_weight=3.0, gray_weight=1.0, threshold=0.01, eps=1e-6):
        super(WGLoss, self).__init__()
        self.white_weight = white_weight
        self.gray_weight = gray_weight
        self.mse_loss = nn.MSELoss(reduction='none')
        self.threshold = threshold
        self.eps = eps

    def forward(self, output, target):
        mse_loss = self.mse_loss(output, target)

        output_grad = torch.abs(output[:, :, :-1, :-1] - output[:, :, 1:, 1:])
        target_grad = torch.abs(target[:, :, :-1, :-1] - target[:, :, 1:, 1:])

        grad_loss = self.mse_loss(output_grad, target_grad)

        white_mask = (target[:, :, :-1, :-1] > self.threshold)
        gray_mask = (target[:, :, :-1, :-1] <= self.threshold) & (target[:, :, :-1, :-1] > 0)

        weighted_mse_loss = self.white_weight * mse_loss[:, :, :-1, :-1] * white_mask.float() + \
                            self.gray_weight * mse_loss[:, :, :-1, :-1] * gray_mask.float()

        weighted_grad_loss = self.white_weight * grad_loss * white_mask.float() + \
                             self.gray_weight * grad_loss * gray_mask.float()

        return weighted_mse_loss.mean() + weighted_grad_loss.mean() + self.eps
