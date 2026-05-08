import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, use_bn=False, dropout=0.0):
        super().__init__()
        layers = []

        # First conv
        layers.append(nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=not use_bn))
        if use_bn:
            layers.append(nn.BatchNorm2d(out_ch))
        layers.append(nn.ReLU(inplace=True))

        # Second conv
        layers.append(nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=not use_bn))
        if use_bn:
            layers.append(nn.BatchNorm2d(out_ch))
        layers.append(nn.ReLU(inplace=True))

        if dropout > 0:
            layers.append(nn.Dropout2d(p=dropout))

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class UpBlock(nn.Module):

    def __init__(self, in_ch, out_ch, use_bn=False, dropout=0.0):
        super().__init__()
        self.up   = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = ConvBlock(out_ch, out_ch, use_bn=use_bn, dropout=dropout)

    def forward(self, x):
        return self.conv(self.up(x))

class UpBlockSkip(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch, use_bn=True, dropout=0.0):
        super().__init__()
        self.up   = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = ConvBlock(out_ch + skip_ch, out_ch, use_bn=use_bn, dropout=dropout)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)

class SODBaseline(nn.Module):
    def __init__(self, input_size: int = 128):
        super().__init__()
        self.input_size = input_size

        self.enc1 = ConvBlock(3,   32)
        self.enc2 = ConvBlock(32,  64)
        self.enc3 = ConvBlock(64,  128)
        self.enc4 = ConvBlock(128, 256)
        self.pool = nn.MaxPool2d(2, 2)
        
        self.bottleneck = ConvBlock(256, 512) 
        self.dec4 = UpBlock(512, 256)
        self.dec3 = UpBlock(256, 128)
        self.dec2 = UpBlock(128,  64)
        self.dec1 = UpBlock( 64,  32)           
        self.out_conv = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x);              p1 = self.pool(e1)
        e2 = self.enc2(p1);             p2 = self.pool(e2)
        e3 = self.enc3(p2);             p3 = self.pool(e3)
        e4 = self.enc4(p3);             p4 = self.pool(e4)

        b  = self.bottleneck(p4)
        d4 = self.dec4(b)
        d3 = self.dec3(d4)
        d2 = self.dec2(d3)
        d1 = self.dec1(d2)

        return torch.sigmoid(self.out_conv(d1))

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

class SODImproved(nn.Module):
    def __init__(self, input_size: int = 128, dropout: float = 0.2):
        super().__init__()
        self.input_size = input_size

        self.enc1 = ConvBlock(3,    64, use_bn=True)
        self.enc2 = ConvBlock(64,  128, use_bn=True)
        self.enc3 = ConvBlock(128, 256, use_bn=True)
        self.enc4 = ConvBlock(256, 512, use_bn=True)
        self.pool = nn.MaxPool2d(2, 2)

        self.bottleneck = ConvBlock(512, 1024, use_bn=True, dropout=dropout)
        self.dec4 = UpBlockSkip(1024, 512, 512, use_bn=True)
        self.dec3 = UpBlockSkip( 512, 256, 256, use_bn=True)
        self.dec2 = UpBlockSkip( 256, 128, 128, use_bn=True)
        self.dec1 = UpBlockSkip( 128,  64,  64, use_bn=True)
        
        self.out_conv = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x);  p1 = self.pool(e1)
        e2 = self.enc2(p1); p2 = self.pool(e2)
        e3 = self.enc3(p2); p3 = self.pool(e3)
        e4 = self.enc4(p3); p4 = self.pool(e4)

        b  = self.bottleneck(p4)

        d4 = self.dec4(b,  e4)
        d3 = self.dec3(d4, e3)
        d2 = self.dec2(d3, e2)
        d1 = self.dec1(d2, e1)

        return torch.sigmoid(self.out_conv(d1))

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

class SODLoss(nn.Module):
    def __init__(self, bce_weight: float = 1.0, iou_weight: float = 0.5, smooth: float = 1e-6):
        super().__init__()
        self.bce_weight = bce_weight
        self.iou_weight = iou_weight
        self.smooth     = smooth
        self.bce        = nn.BCELoss()

    def iou_loss(self, pred, target):
        pred_flat   = pred.view(-1)
        target_flat = target.view(-1)
        intersection = (pred_flat * target_flat).sum()
        union        = pred_flat.sum() + target_flat.sum() - intersection
        iou          = (intersection + self.smooth) / (union + self.smooth)
        return 1.0 - iou

    def forward(self, pred, target):
        bce  = self.bce(pred, target)
        iou  = self.iou_loss(pred, target)
        loss = self.bce_weight * bce + self.iou_weight * iou
        return loss, bce.item(), iou.item()

def build_model(variant: str = "baseline", input_size: int = 128, **kwargs) -> nn.Module:
    if variant == "baseline":
        model = SODBaseline(input_size=input_size)
    elif variant == "improved":
        model = SODImproved(input_size=input_size, **kwargs)
    else:
        raise ValueError(f"Unknown variant '{variant}'. Choose 'baseline' or 'improved'.")

    print(f"[build_model] {variant.upper()}  |  params = {model.count_params():,}")
    return model

if __name__ == "__main__":
    for variant in ("baseline", "improved"):
        model = build_model(variant, input_size=128)
        x     = torch.randn(2, 3, 128, 128)
        y     = model(x)
        print(f"  {variant}: input {x.shape} → output {y.shape}  "
              f"(min={y.min():.3f}, max={y.max():.3f})")

    criterion = SODLoss()
    pred   = torch.sigmoid(torch.randn(2, 1, 128, 128))
    target = (torch.rand(2, 1, 128, 128) > 0.5).float()
    loss, bce, iou = criterion(pred, target)
    print(f"\nLoss test  total={loss:.4f}  bce={bce:.4f}  iou_loss={iou:.4f}")
    print("\nsod_model.py self-test passed ✓")
