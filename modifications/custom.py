# Modified custom.py to include APA-Coupled Focal Loss

import torch.nn as nn

class APAFocalLoss(nn.Module):
    def __init__(self, gamma=2.0):
        super(APAFocalLoss, self).__init__()
        self.gamma = gamma

    def forward(self, logits, targets, kappa, lambda_):
        # Convert targets to one-hot if needed
        if targets.dim() == 1 or (targets.dim() == 2 and targets.size(1) == 1):
            targets = torch.nn.functional.one_hot(targets, num_classes=logits.size(1)).float()
        probs = torch.sigmoid(logits)
        focal_weight = (1 - probs) ** self.gamma
        adaptive_weight = torch.exp(-torch.abs(kappa * logits - lambda_))
        loss = -adaptive_weight * focal_weight * targets * torch.log(probs + 1e-8)
        return loss.mean()