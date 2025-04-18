# Modified custom.py to include APA-Coupled Focal Loss
import numpy as np
import torch.nn.functional as F
import torch
import torch.nn as nn


class LDAMLoss(nn.Module):
    def __init__(self, cls_num_list, max_m=0.5, weight=None, s=30):
        super(LDAMLoss, self).__init__()
        m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
        m_list = m_list * (max_m / np.max(m_list))
        self.m_list = torch.tensor(m_list, dtype=torch.float32)
        self.s = s
        self.weight = weight

    def forward(self, x, target):
        if x.device != self.m_list.device:
            self.m_list = self.m_list.to(x.device)

        index = torch.zeros_like(x, dtype=torch.uint8)
        index.scatter_(1, target.view(-1, 1), 1)
        index_float = index.type(torch.float32)

        batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0, 1)).view(-1, 1)
        x_m = x - index_float * batch_m
        output = self.s * x_m
        return F.cross_entropy(output, target, weight=self.weight)
