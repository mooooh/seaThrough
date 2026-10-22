"""ResNet-18 encoder with a skip-connected decoder for dense height regression."""
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import ResNet18_Weights, resnet18


class ConvBlock(nn.Sequential):
    def __init__(self, cin: int, cout: int):
        super().__init__(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False),
            nn.GroupNorm(8, cout), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False),
            nn.GroupNorm(8, cout), nn.ReLU(inplace=True))


class UpBlock(nn.Module):
    def __init__(self, cin: int, skip: int, cout: int):
        super().__init__()
        self.conv = ConvBlock(cin + skip, cout)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


class WaveHeightNet(nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()
        backbone = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)
        self.pool = backbone.maxpool
        self.layer1, self.layer2 = backbone.layer1, backbone.layer2
        self.layer3, self.layer4 = backbone.layer3, backbone.layer4
        self.up3, self.up2 = UpBlock(512, 256, 128), UpBlock(128, 128, 64)
        self.up1, self.up0 = UpBlock(64, 64, 32), UpBlock(32, 64, 32)
        self.head = nn.Sequential(ConvBlock(32, 16), nn.Conv2d(16, 1, 1))

    def train(self, mode: bool = True):
        super().train(mode)
        # Preserve pretrained running statistics; batch size can be small on an 8GB GPU.
        # Affine parameters still learn. GroupNorm is used in the new decoder.
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.shape[-2:]
        x0 = self.stem(x)
        x1 = self.layer1(self.pool(x0))
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        value = self.up0(self.up1(self.up2(self.up3(x4, x3), x2), x1), x0)
        value = F.interpolate(value, size=size, mode="bilinear", align_corners=False)
        return torch.sigmoid(self.head(value))

