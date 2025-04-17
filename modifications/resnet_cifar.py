import torch.nn as nn
import torch.nn.functional as F
import torch
import math

# This file contains the implementation of ResNet architectures specifically designed for CIFAR datasets.
# Modified to include Adaptive Parametric Activation (APA) enhancements for CIFAR-100 LT.

class AdaptiveParametricActivation(nn.Module):
    def __init__(self):
        super(AdaptiveParametricActivation, self).__init__()
        self.alpha = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return self.alpha * x + self.beta

# Add APA enhancements for ResNet20
class BasicBlockAPA(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlockAPA, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.apa = AdaptiveParametricActivation()  # APA with learned parameters
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = self.apa(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

# Modify ResNet20 to use BasicBlockAPA
class ResNet20APA(nn.Module):
    def __init__(self, num_classes=100):
        super(ResNet20APA, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(BasicBlockAPA, 16, 3, stride=1)
        self.layer2 = self._make_layer(BasicBlockAPA, 32, 3, stride=2)
        self.layer3 = self._make_layer(BasicBlockAPA, 64, 3, stride=2)
        self.linear = nn.Linear(256, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

# Add APA-specific ResNet20 to the module exports
def resnet20_apa(num_classes=100):
    return ResNet20APA(num_classes=num_classes)