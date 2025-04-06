# resnet_pytorch_modified.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Type, Any, Callable, Union, List, Optional

class HybridAPA(nn.Module):
    """Combines APA activation with AdAct's piecewise linear components"""
    def __init__(self, num_parameters=1):
        super().__init__()
        self.kappa = nn.Parameter(torch.ones(num_parameters))
        self.lambda_ = nn.Parameter(torch.ones(num_parameters))
        self.hinges = nn.Parameter(torch.tensor([0.2, 0.5, 0.8]))  # Learnable hinges
        
    def forward(self, x):
        # APA component
        apa = (self.lambda_ * torch.exp(-self.kappa * x) + 1).pow(-1/self.lambda_)
        # AdAct component
        piecewise = sum([F.relu(x - h) for h in self.hinges])
        return apa + piecewise

class SE_Block(nn.Module):
    """Modified SE Block with frequency conditioning and HybridAPA"""
    def __init__(self, c, r=16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(c, c // r),
            nn.ReLU(),
            nn.Linear(c // r, c),
            HybridAPA(num_parameters=c)  # Replaced Sigmoid with HybridAPA
        )
        # Frequency conditioning components
        self.fft_adapter = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, c)
        )

    def forward(self, x):
        bs, c, _, _ = x.shape
        # Frequency domain processing
        fft = torch.fft.fft2(x)
        mag = torch.abs(fft).mean(dim=(2,3))
        freq_weights = self.fft_adapter(mag)
        
        # Channel attention with frequency conditioning
        y = self.squeeze(x).view(bs, c)
        y = self.excitation(y + freq_weights).view(bs, c, 1, 1)
        return x * y

class NeuralCollapseRegularizer(nn.Module):
    """Added regularization for feature stability"""
    def __init__(self):
        super().__init__()
        
    def forward(self, features, labels):
        unique_labels = torch.unique(labels)
        class_means = torch.stack([features[labels==lbl].mean(0) for lbl in unique_labels])
        global_mean = features.mean(0)
        
        # Within-class covariance
        S_W = sum([(features[labels==lbl] - class_means[i]).T @ 
                  (features[labels==lbl] - class_means[i]) 
                  for i, lbl in enumerate(unique_labels)])
        
        # Between-class covariance
        S_B = sum([(class_means[i] - global_mean).T @ 
                  (class_means[i] - global_mean) 
                  for i in range(len(unique_labels))])
        
        return torch.trace(S_W) / torch.trace(S_B)

def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride,
        padding=dilation, groups=groups, bias=False, dilation=dilation
    )

def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class Bottleneck(nn.Module):
    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.se = SE_Block(planes * self.expansion)  # Modified SE Block

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out = self.se(out)  # Apply modified SE Block

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ResNet(nn.Module):
    def __init__(
        self,
        block: Type[Union[Bottleneck]],
        layers: List[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
            
        self._norm_layer = norm_layer
        self.inplanes = 64
        self.dilation = 1
        self.groups = groups
        self.base_width = width_per_group
        self.nc_regularizer = NeuralCollapseRegularizer()

        # Initial layers
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Residual layers
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # Final layers
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # Initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)

    def _make_layer(
        self,
        block: Type[Bottleneck],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation

        if dilate:
            self.dilation *= stride
            stride = 1
            
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(
            self.inplanes, planes, stride, downsample, self.groups,
            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        
        for _ in range(1, blocks):
            layers.append(block(
                self.inplanes, planes, groups=self.groups,
                base_width=self.base_width, dilation=self.dilation,
                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x: Tensor, labels: Optional[Tensor] = None) -> Tensor:
        x = self._forward_impl(x)
        if self.training and labels is not None:
            return x, self.nc_regularizer(x, labels)
        return x

def se_resnet50_apa(**kwargs: Any) -> ResNet:
    return ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
