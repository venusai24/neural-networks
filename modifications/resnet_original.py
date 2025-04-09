import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

__all__ = ['ResNet', 'resnet20_orig', 'resnet32_orig', 'resnet44_orig', 'resnet56_orig', 'resnet110_orig', 'resnet1202_orig', 'se_resnet32_orig']

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A', use_se=False):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes // 4, planes // 4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(self.expansion * planes)
                )

        self.use_se = use_se
        if self.use_se:
            self.se = SEBlock(planes)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.use_se:
            out = self.se(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.fc1 = nn.Linear(channels, channels // reduction, bias=False)
        self.fc2 = nn.Linear(channels // reduction, channels, bias=False)

    def forward(self, x):
        b, c, _, _ = x.size()
        y = F.adaptive_avg_pool2d(x, 1).view(b, c)
        y = F.relu(self.fc1(y))  # ReLU replaces AGLU
        y = torch.sigmoid(self.fc2(y))  # Sigmoid replaces APA
        y = y.view(b, c, 1, 1)
        return x * y.expand_as(x)

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, use_se=False):
        super(ResNet, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1, use_se=use_se)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2, use_se=use_se)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2, use_se=use_se)
        self.linear = nn.Linear(64, num_classes)

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride, use_se):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, use_se=use_se))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def _weights_init(m):
    classname = m.__class__.__name__
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)

def resnet20_orig(num_classes = 100):
    return ResNet(BasicBlock, [3, 3, 3], num_classes=num_classes)

def resnet32_orig(num_classes=10):
    return ResNet(BasicBlock, [5, 5, 5], num_classes=num_classes)

def resnet44_orig():
    return ResNet(BasicBlock, [7, 7, 7])

def resnet56_orig():
    return ResNet(BasicBlock, [9, 9, 9])

def resnet110_orig():
    return ResNet(BasicBlock, [18, 18, 18])

def resnet1202_orig():
    return ResNet(BasicBlock, [200, 200, 200])

def se_resnet32_orig(num_classes=10):
    return ResNet(BasicBlock, [5, 5, 5], num_classes=num_classes, use_se=True)

if __name__ == "__main__":
    # Test the SE-ResNet implementation
    net = se_resnet32_orig(num_classes=100)
    y = net(torch.randn(1, 3, 32, 32))
    print(y.size())