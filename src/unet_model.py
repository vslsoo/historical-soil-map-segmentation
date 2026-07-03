"""A small U-Net for 4-class (background + class_10/12/13) map segmentation."""
import torch
import torch.nn as nn


def conv_block(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class UNet(nn.Module):
    def __init__(self, in_channels: int = 3, num_classes: int = 4, base_channels: int = 32):
        super().__init__()
        c1, c2, c3, c4 = base_channels, base_channels * 2, base_channels * 4, base_channels * 8

        self.enc1 = conv_block(in_channels, c1)
        self.enc2 = conv_block(c1, c2)
        self.enc3 = conv_block(c2, c3)
        self.bottleneck = conv_block(c3, c4)

        self.pool = nn.MaxPool2d(2)

        self.up3 = nn.ConvTranspose2d(c4, c3, 2, stride=2)
        self.dec3 = conv_block(c4, c3)
        self.up2 = nn.ConvTranspose2d(c3, c2, 2, stride=2)
        self.dec2 = conv_block(c3, c2)
        self.up1 = nn.ConvTranspose2d(c2, c1, 2, stride=2)
        self.dec1 = conv_block(c2, c1)

        self.out_conv = nn.Conv2d(c1, num_classes, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))

        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return self.out_conv(d1)


def build_model(num_classes: int, architecture: str = "custom", encoder_name: str = "resnet34",
                 encoder_weights: str = "imagenet") -> nn.Module:
    """architecture="custom" is the from-scratch UNet above; architecture="smp" uses
    segmentation_models_pytorch with an ImageNet-pretrained encoder, which tends to
    generalize better with small training sets since it already knows general-purpose
    edge/texture features instead of learning them from a handful of patches."""
    if architecture == "custom":
        return UNet(in_channels=3, num_classes=num_classes)
    if architecture == "smp":
        import segmentation_models_pytorch as smp
        return smp.Unet(encoder_name=encoder_name, encoder_weights=encoder_weights,
                         in_channels=3, classes=num_classes)
    raise ValueError(f"Unknown architecture: {architecture!r} (expected 'custom' or 'smp')")
