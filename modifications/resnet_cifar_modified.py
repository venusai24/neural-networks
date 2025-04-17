import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn import Parameter
import math

__all__ = ['ResNet_s', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110', 'resnet1202']

class HybridAPA(nn.Module):
    """Integrated APA + AdAct + Frequency-Conditioned Activation"""
    def __init__(self, num_parameters=1):
        super().__init__()
        # APA parameters
        self.kappa = nn.Parameter(torch.ones(num_parameters))
        self.lambda_ = nn.Parameter(torch.ones(num_parameters))
        
        # AdAct parameters
        self.hinges = nn.Parameter(torch.tensor([0.2, 0.5, 0.8]))
        
        # Frequency conditioning
        self.fft_adapter = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, 3)  # Predicts 3 hinge positions
        )

    def forward(self, x):
        # Frequency adaptation
        fft = torch.fft.rfft2(x, norm='ortho')
        mag = torch.abs(fft).mean(dim=(2,3))
        hinges = self.fft_adapter(mag.unsqueeze(-1)).squeeze()
        
        # APA component
        apa = (self.lambda_ * torch.exp(-self.kappa * x) + 1).pow(-1/self.lambda_)
        
        # AdAct component with dynamic hinges
        piecewise = sum([F.relu(x - h) for h in hinges])
        
        return apa + piecewise

class AdaptiveParametricActivation(nn.Module):
    def forward(self, x):
        self.kappa.data.clamp_(0.5, 5.0)
        self.lambda_.data.clamp_(0.1, 2.0)
        return super().forward(x)

class ClassAwareAttention(nn.Module):
    def __init__(self, num_classes, in_channels):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Conv2d(in_channels, num_classes, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x, class_idx):
        attention_map = self.attention(x)
        return x * attention_map[class_idx].unsqueeze(2).unsqueeze(3)

def _weights_init(m):
    classname = m.__class__.__name__
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super().__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)

class BasicBlock(nn.Module):
    expansion = 1  # Add this attribute

    def __init__(self, in_planes, planes):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1)
        self.apa = AdaptiveParametricActivation()
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1)
        
    def forward(self, x):
        identity = x
        out = self.apa(self.conv1(x))
        out = self.conv2(out)
        out += identity
        return out

class ResNet_s(nn.Module):
    def __init__(self, block, num_blocks, num_classes=100, use_norm=None, use_gumbel=False):
        super().__init__()
        self.in_planes = 16
        self.use_gumbel = use_gumbel
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.activation = HybridAPA() if use_gumbel else nn.ReLU()
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        if use_norm == 'cosine':
            self.linear = nn.utils.weight_norm(nn.Linear(64, num_classes), dim=1)
        else:
            self.linear = nn.Linear(64, num_classes)
        self.apply(_weights_init)
        
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = self.activation(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def resnet20(num_classes=100, use_norm=None, use_gumbel=False, use_gumbel_cb=False):
    return ResNet_s(
        BasicBlock, 
        [3, 3, 3], 
        num_classes=num_classes, 
        use_norm=use_norm,
        use_gumbel=use_gumbel
    )

def test(net):
    total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params/1e6:.2f}M")
    print("Architecture:\n", net)

if __name__ == "__main__":
    # Example usage
    model = resnet20(num_classes=100, use_gumbel=True)
    test(model)
