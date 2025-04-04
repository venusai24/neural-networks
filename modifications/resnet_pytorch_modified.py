# resnet_pytorch.py:
# This file contains the implementation of ResNet architectures for PyTorch.
# It includes various ResNet variants with additional attention mechanisms like SE_Block and CBAM, 
# as well as initialization functions for these models.

from typing import Type, Any, Callable, Union, List, Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import Parameter
try:
    import resnet_cifar
    import cbam 
except ImportError:
    from classification import resnet_cifar
    from classification import cbam 
try:
    from torch.hub import load_state_dict_from_url  # noqa: 401
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url


def _download_file_from_remote_location(fpath: str, url: str) -> None:
    pass

__all__ = [
    "ResNet",
    "resnet18",
    "resnet34",
    "resnet50",
    "resnet101",
    "resnet152",
    "resnext50_32x4d",
    "resnext101_32x8d",
    "wide_resnet50_2",
    "wide_resnet101_2",
]


model_urls = {
    "resnet18": "https://download.pytorch.org/models/resnet18-f37072fd.pth",
    "resnet34": "https://download.pytorch.org/models/resnet34-b627a593.pth",
    "resnet50": "https://download.pytorch.org/models/resnet50-0676ba61.pth",
    "resnet101": "https://download.pytorch.org/models/resnet101-63fe2227.pth",
    "resnet152": "https://download.pytorch.org/models/resnet152-394f9c45.pth",
    "resnext50_32x4d": "https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth",
    "resnext101_32x8d": "https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth",
    "wide_resnet50_2": "https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth",
    "wide_resnet101_2": "https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth",
}

class Unified(nn.Module):
    __constants__ = ['num_parameters']
    num_parameters: int

    def __init__(self, num_parameters: int = 1,device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.num_parameters = num_parameters
        super().__init__()
        lambda_param = torch.nn.init.uniform_(torch.empty(num_parameters, **factory_kwargs))
        kappa_param = torch.nn.init.uniform_(torch.empty(num_parameters, **factory_kwargs),a=-1.0,b=0.0) #works well with this init
        self.softplus = nn.Softplus(beta=-1.0)
        
        if num_parameters>1:
            self.lambda_param = nn.Parameter(lambda_param.unsqueeze(0).unsqueeze(-1).unsqueeze(-1))
            self.kappa_param = nn.Parameter(kappa_param.unsqueeze(0).unsqueeze(-1).unsqueeze(-1))
        else:
            self.lambda_param = nn.Parameter(lambda_param)
            self.kappa_param = nn.Parameter(kappa_param)

    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        l = torch.clamp(self.lambda_param,min=0.0001)
        p = torch.exp((1/l) * self.softplus((self.kappa_param*input) - torch.log(l)))
        
        return p

    def extra_repr(self) -> str:
        return 'num_parameters={}'.format(self.num_parameters)
    


class UniRect_static(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, lambda_param, kappa_param):
        # normal forward pass
#         input, lambda_param, kappa_param = input.detach(), lambda_param.detach(), kappa_param.detach()
        p  = torch.exp((1/lambda_param) * torch.nn.functional.softplus(kappa_param*input -torch.log(lambda_param), beta=-1.0, threshold=20))
        ctx.save_for_backward(input, lambda_param, kappa_param,p)
        out = p * input

        return out

    @staticmethod
    def backward(ctx, grad_output):
        input, lambda_param, kappa_param, p = ctx.saved_tensors
        sigmoidal_coeff = p/(lambda_param + torch.exp(kappa_param*input))

        part_grad_kappa = (input**2)*sigmoidal_coeff
        part_grad_lambda = (-input/lambda_param)*sigmoidal_coeff
        part_grad_input = kappa_param*input*sigmoidal_coeff + p

        grad_input = grad_output*part_grad_input
        grad_lambda = grad_output*part_grad_lambda
        grad_kappa = grad_output*part_grad_kappa

        return grad_input, grad_lambda, grad_kappa


class UniRect(nn.Module):
    __constants__ = ['num_parameters']
    num_parameters: int

    def __init__(self, num_parameters: int = 1,lambda_init=(0.0,1.0),kappa_init=(0.8,1.2),device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.num_parameters = num_parameters
        super().__init__()
        lambda_param = torch.nn.init.uniform_(torch.empty(num_parameters, **factory_kwargs),a=lambda_init[0],b=lambda_init[1])
        kappa_param = torch.nn.init.uniform_(torch.empty(num_parameters, **factory_kwargs),a=kappa_init[0],b=kappa_init[1])

        if num_parameters>1:
            self.lambda_param = nn.Parameter(lambda_param.unsqueeze(0).unsqueeze(-1).unsqueeze(-1))
            self.kappa_param = nn.Parameter(kappa_param.unsqueeze(0).unsqueeze(-1).unsqueeze(-1))
        else:
            self.lambda_param = nn.Parameter(lambda_param)
            self.kappa_param = nn.Parameter(kappa_param)

        self.relu = torch.nn.ReLU()
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        lambda_param = self.relu(self.lambda_param) + 1e-8
        out = UniRect_static.apply(input, lambda_param, self.kappa_param)

        return out

    def extra_repr(self) -> str:
        return 'num_parameters={}'.format(self.num_parameters)


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion: int = 1

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
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
    

class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

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
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

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

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        

        return out


class ResNet(nn.Module):
    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        use_norm: str = None,
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
        self.dual_head=False
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        
        if (block.__name__.endswith('Gumbel')) or (block.__name__.endswith('GumbelSigmoid')) is True:
            self.relu = UniRect()
        else:
            self.relu = nn.ReLU(inplace=True)
            
            
        
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        if use_norm=='cosine':
            self.fc = resnet_cifar.CosNorm_Classifier(512 * block.expansion, num_classes)
        elif use_norm=='inv':
            self.fc = resnet_cifar.InvCosNorm_Classifier(512 * block.expansion, num_classes)
        elif use_norm=='lr_cosine':
            self.fc = resnet_cifar.CosNorm_Classifier(512 * block.expansion, num_classes,learnable=True)
        elif use_norm=='norm':
            self.fc = resnet_cifar.NormedLinear(512 * block.expansion, num_classes)
        else:
            self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
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
        layers.append(
            block(
                self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
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

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


class ResNet_TwoBranch(nn.Module):
    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        use_norm: str = None,
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
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        if use_norm=='cosine':
            self.fc1 = resnet_cifar.CosNorm_Classifier(512 * block.expansion, num_classes)
            self.fc2 = resnet_cifar.CosNorm_Classifier(512 * block.expansion, num_classes)
        elif use_norm=='norm':
            self.fc1 = resnet_cifar.NormedLinear(512 * block.expansion, num_classes)
            self.fc2 = resnet_cifar.NormedLinear(512 * block.expansion, num_classes)
        else:
            self.fc1 = nn.Linear(512 * block.expansion, num_classes)
            self.fc2 = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
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
        layers.append(
            block(
                self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
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
        logits1 = self.fc1(x)
        logits2 = self.fc2(x)


        return logits1,logits2

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)
    

class SE_Block(nn.Module):
    "credits: https://github.com/moskomule/senet.pytorch/blob/master/senet/se_module.py#L4"
    def __init__(self, c, r=16,use_gumbel=False):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.use_gumbel = use_gumbel
        
        if self.use_gumbel is True:
            self.norm = nn.LayerNorm(c)
            self.uniact=Unified()
            self.excitation = nn.Sequential(
                nn.Linear(c, c // r, bias=False),
                nn.ReLU(inplace=True),
                nn.Linear(c // r, c, bias=False),
                nn.Dropout(p=0.1)
            )
        else:
            self.excitation = nn.Sequential(
                nn.Linear(c, c // r, bias=False),
                nn.ReLU(inplace=True),
                nn.Linear(c // r, c, bias=False)
            )

    def forward(self, x):
        bs, c, _, _ = x.shape
        y = self.squeeze(x).view(bs, c)
        if self.use_gumbel is True:
            y= self.norm(y)
        y = self.excitation(y).view(bs, c, 1, 1)
        if self.use_gumbel is True:
            y = self.uniact(y)
        else:
            y=y.sigmoid()
                        
        return x * y.expand_as(x)


class SEBottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

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
        r:int = 16,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        # add SE block
        self.se = SE_Block(planes * self.expansion, r)

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

        # Add SE operation
        out = self.se(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
    
class SEBottleneckGumbel(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

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
        r:int = 16,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.unirect1 = UniRect()
        self.unirect2 = UniRect()
        self.unirect3 = UniRect()
        self.downsample = downsample
        self.stride = stride
        # add SE block
        self.se = SE_Block(planes * self.expansion, r,use_gumbel=True)

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.unirect1(out)
        

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.unirect2(out)


        out = self.conv3(out)
        out = self.bn3(out)

        # Add SE operation
        out = self.se(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.unirect3(out)


        return out
    

class CBAMBottleneckGumbel(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

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
        r:int = 16,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.unirect1 = UniRect()
        self.unirect2 = UniRect()
        self.unirect3 = UniRect()
        self.downsample = downsample
        self.stride = stride
        # add SE block
        self.cbam = cbam.CBAM(planes * self.expansion, r,use_gumbel=True,use_gumbel_cb=True)

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.unirect1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.unirect2(out)

        out = self.conv3(out)
        out = self.bn3(out)

        # Add SE operation
        out = self.cbam(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
#         out = self.relu(out)
        out = self.unirect3(out)
        

        return out
    
class SpatialBottleneckGumbel(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

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
        r:int = 16,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.unirect1 = UniRect()
        self.unirect2 = UniRect()
        self.unirect3 = UniRect()
        self.downsample = downsample
        self.stride = stride
        # add SE block
        self.cbam = cbam.CBAM(planes * self.expansion, r,use_gumbel=True,use_gumbel_cb=True,no_channel=True)

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.unirect1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.unirect2(out)

        out = self.conv3(out)
        out = self.bn3(out)

        # Add SE operation
        out = self.cbam(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.unirect3(out)
        

        return out

class SpatialBottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

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
        r:int = 16,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        # add SE block
        self.cbam = cbam.CBAM(planes * self.expansion, r,use_gumbel=False,use_gumbel_cb=False,no_channel=True)

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

        # Add SE operation
        out = self.cbam(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
    
class CBAMBottleneckGumbelSigmoid(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

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
        r:int = 16,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.unirect1 = UniRect()
        self.unirect2 = UniRect()
        self.unirect3 = UniRect()
        self.downsample = downsample
        self.stride = stride
        # add SE block
        self.cbam = cbam.CBAM(planes * self.expansion, r,use_gumbel=True,use_gumbel_cb=False)

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.unirect1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.unirect2(out)

        out = self.conv3(out)
        out = self.bn3(out)

        # Add SE operation
        out = self.cbam(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.unirect3(out)

        return out
    
class CBAMBottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

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
        r:int = 16,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        # add SE block
        self.cbam = cbam.CBAM(planes * self.expansion, r,use_gumbel=False)

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

        # Add SE operation
        out = self.cbam(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

def _mismatched_classifier(model,arch,pretrained):
    classifier_name, old_classifier = model._modules.popitem()
    classifier_input_size = old_classifier.in_features
    pretrained_classifier = nn.Linear(classifier_input_size, 1000)
    model.add_module(classifier_name, pretrained_classifier)
    if pretrained =='pytorch':
        state_dict = load_state_dict_from_url(model_urls[arch], progress=False)
        model.load_state_dict(state_dict)
    else:
        state_dict = torch.load(pretrained, map_location='cpu')
        model.load_state_dict(state_dict['model'],strict=False)

    classifier_name, new_classifier = model._modules.popitem()
    model.add_module(classifier_name, old_classifier)
    return model



def _resnet(
    arch: str,
    block: Type[Union[BasicBlock, Bottleneck]],
    layers: List[int],
    pretrained: str,
    progress: bool,
    use_norm: str,
    **kwargs: Any,
) -> ResNet:
    model = ResNet(block, layers,use_norm, **kwargs)
    if pretrained!='None':
        if kwargs['num_classes']!=1000:
            model = _mismatched_classifier(model,arch,pretrained)
        else:
            state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
            model.load_state_dict(state_dict)
    return model

def _tb_resnet(
    arch: str,
    block: Type[Union[BasicBlock, Bottleneck]],
    layers: List[int],
    pretrained: str,
    progress: bool,
    use_norm: str,
    **kwargs: Any,
) -> ResNet:
    model = ResNet_TwoBranch(block, layers,use_norm, **kwargs)
    return model


def resnet18(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet("resnet18", BasicBlock, [2, 2, 2, 2], pretrained, progress, **kwargs)


def resnet34(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet("resnet34", BasicBlock, [3, 4, 6, 3], pretrained, progress, **kwargs)

def resnet50(pretrained: str = None, progress: bool = True,use_norm: str = None,use_gumbel=False,use_gumbel_cb=False, **kwargs: Any) -> ResNet:
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet("resnet50", Bottleneck, [3, 4, 6, 3], pretrained, progress,use_norm=use_norm, **kwargs)


def tb_resnet50(pretrained: str = None, progress: bool = True,use_norm: str = None, **kwargs: Any) -> ResNet:

    return _tb_resnet("resnet50", Bottleneck, [3, 4, 6, 3], pretrained, progress,use_norm=use_norm, **kwargs)

def resnet101(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-101 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet("resnet101", Bottleneck, [3, 4, 23, 3], pretrained, progress, **kwargs)

def se_resnet101(pretrained: bool = False, progress: bool = True,use_norm: str = None,use_gumbel=False,use_gumbel_cb=False, **kwargs: Any) -> ResNet:
    r"""ResNet-101 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    if use_gumbel is True:
        return _resnet("resnet101", SEBottleneckGumbel, [3, 4, 23, 3], pretrained, progress,use_norm=use_norm, **kwargs)
    else:
        return _resnet("resnet101", SEBottleneck, [3, 4, 23, 3], pretrained, progress,use_norm=use_norm, **kwargs)


def resnet152(pretrained: str = None, progress: bool = True,use_norm: str = None,use_gumbel=False,use_gumbel_cb=False,  **kwargs: Any) -> ResNet:
    r"""ResNet-152 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet("resnet152", Bottleneck, [3, 8, 36, 3], pretrained, progress,use_norm=use_norm, **kwargs)


def se_resnet152(pretrained: str = None, progress: bool = True,use_norm: str = None,use_gumbel=False,use_gumbel_cb=False,  **kwargs: Any) -> ResNet:
    r"""ResNet-152 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    if use_gumbel is True: 
        return _resnet('resnet152', SEBottleneckGumbel, [3, 8, 36, 3], pretrained, progress,use_norm=use_norm,
                       **kwargs)
    else:
        return _resnet('resnet152', SEBottleneck, [3, 8, 36, 3], pretrained, progress,use_norm=use_norm,
                       **kwargs)

def cb_resnet152(pretrained: str = None, progress: bool = True,use_norm: str = None,use_gumbel=False,use_gumbel_cb=False,  **kwargs: Any) -> ResNet:
    r"""ResNet-152 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    if use_gumbel is True:
        if use_gumbel_cb is False:
            return _resnet("resnet152", CBAMBottleneckGumbelSigmoid, [3, 8, 36, 3], pretrained, progress,use_norm=use_norm, **kwargs)
        else:
            return _resnet("resnet152", CBAMBottleneckGumbel, [3, 8, 36, 3], pretrained, progress,use_norm=use_norm, **kwargs)
    else:
        return _resnet("resnet152", CBAMBottleneck, [3, 8, 36, 3], pretrained, progress,use_norm=use_norm, **kwargs)


def resnext50_32x4d(pretrained: str = None, progress: bool = True,use_norm: str = None,use_gumbel=False,use_gumbel_cb=False, **kwargs: Any) -> ResNet:
    r"""ResNeXt-50 32x4d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs["groups"] = 32
    kwargs["width_per_group"] = 4
    return _resnet("resnext50_32x4d", Bottleneck, [3, 4, 6, 3], pretrained, progress,use_norm=use_norm, **kwargs)

def se_resnext50_32x4d(pretrained: str = None, progress: bool = True,use_norm: str = None,use_gumbel=False,use_gumbel_cb=False, **kwargs: Any) -> ResNet:
    r"""ResNeXt-50 32x4d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs["groups"] = 32
    kwargs["width_per_group"] = 4
    
    if use_gumbel is True: 
        return _resnet('resnext50_32x4d', SEBottleneckGumbel, [3, 4, 6, 3], pretrained, progress,use_norm=use_norm,
                       **kwargs)
    else:
        return _resnet('resnext50_32x4d', SEBottleneck, [3, 4, 6, 3], pretrained, progress,use_norm=use_norm,
                       **kwargs)
    
def cb_resnext50_32x4d(pretrained: str = None, progress: bool = True,use_norm: str = None,use_gumbel=False,use_gumbel_cb=False, **kwargs: Any) -> ResNet:
    r"""ResNeXt-50 32x4d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs["groups"] = 32
    kwargs["width_per_group"] = 4
    
    if use_gumbel is True:
        if use_gumbel_cb is False:
            return _resnet('resnext50_32x4d', CBAMBottleneckGumbelSigmoid, [3, 4, 6, 3], pretrained, progress,use_norm=use_norm,
                           **kwargs)
        else:
            return _resnet('resnext50_32x4d', CBAMBottleneckGumbel, [3, 4, 6, 3], pretrained, progress,use_norm=use_norm,
                           **kwargs)
    else:
        return _resnet('resnext50_32x4d', CBAMBottleneck, [3, 4, 6, 3], pretrained, progress,use_norm=use_norm,
                       **kwargs)

    
def sp_resnext50_32x4d(pretrained: str = None, progress: bool = True,use_norm: str = None,use_gumbel=False,use_gumbel_cb=False, **kwargs: Any) -> ResNet:
    r"""ResNeXt-50 32x4d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs["groups"] = 32
    kwargs["width_per_group"] = 4
    
    if use_gumbel is True:
        return _resnet('resnext50_32x4d', SpatialBottleneckGumbel, [3, 4, 6, 3], pretrained, progress,use_norm=use_norm,
                       **kwargs)
    else:
        return _resnet('resnext50_32x4d', SpatialBottleneck, [3, 4, 6, 3], pretrained, progress,use_norm=use_norm,
                       **kwargs)


def resnext101_32x8d(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNeXt-101 32x8d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs["groups"] = 32
    kwargs["width_per_group"] = 8
    return _resnet("resnext101_32x8d", Bottleneck, [3, 4, 23, 3], pretrained, progress, **kwargs)


def wide_resnet50_2(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""Wide ResNet-50-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_.
    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs["width_per_group"] = 64 * 2
    return _resnet("wide_resnet50_2", Bottleneck, [3, 4, 6, 3], pretrained, progress, **kwargs)


def wide_resnet101_2(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""Wide ResNet-101-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_.
    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs["width_per_group"] = 64 * 2
    return _resnet("wide_resnet101_2", Bottleneck, [3, 4, 23, 3], pretrained, progress, **kwargs)


def se_resnet50(pretrained=None, progress=True,use_norm=None,use_gumbel=False,use_gumbel_cb=False, **kwargs):
    if use_gumbel is True: 
        return _resnet('resnet50', SEBottleneckGumbel, [3, 4, 6, 3], pretrained, progress,use_norm=use_norm,
                       **kwargs)
    else:
        return _resnet('resnet50', SEBottleneck, [3, 4, 6, 3], pretrained, progress,use_norm=use_norm,
                       **kwargs)
    
def cb_resnet50(pretrained=None, progress=True,use_norm=None,use_gumbel=False,use_gumbel_cb=False,**kwargs):
    if use_gumbel is True:
        if use_gumbel_cb is False:
            return _resnet('resnet50', CBAMBottleneckGumbelSigmoid, [3, 4, 6, 3], pretrained, progress,use_norm=use_norm,
                           **kwargs)
        else:
            return _resnet('resnet50', CBAMBottleneckGumbel, [3, 4, 6, 3], pretrained, progress,use_norm=use_norm,
                           **kwargs)
    else:
        return _resnet('resnet50', CBAMBottleneck, [3, 4, 6, 3], pretrained, progress,use_norm=use_norm,
                       **kwargs)
    
    
def sp_resnet50(pretrained=None, progress=True,use_norm=None,use_gumbel=False,use_gumbel_cb=False,**kwargs):
    if use_gumbel is True:
        return _resnet('resnet50', SpatialBottleneckGumbel, [3, 4, 6, 3], pretrained, progress,use_norm=use_norm,
                           **kwargs)
    else:
        return _resnet('resnet50', SpatialBottleneck, [3, 4, 6, 3], pretrained, progress,use_norm=use_norm,
                       **kwargs)
        
