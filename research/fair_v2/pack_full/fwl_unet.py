"""U-Net denoiser: noisy masked kappa map (+ mask channel) -> E[clean kappa | x] on the mask.

BatchNorm (fixed affine in eval mode) rather than per-sample norms, so the
global amplitude of a map is not normalized away; in eval mode the output for
one map never depends on other maps in the batch.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

SCALE = 0.03


class Block(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.c1 = nn.Conv2d(cin, cout, 3, 1, 1, bias=False)
        self.b1 = nn.BatchNorm2d(cout)
        self.c2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.b2 = nn.BatchNorm2d(cout)
        self.skip = nn.Identity() if cin == cout else nn.Conv2d(cin, cout, 1, bias=False)

    def forward(self, x):
        h = F.silu(self.b1(self.c1(x)))
        h = self.b2(self.c2(h))
        return F.silu(h + self.skip(x))


class UNet(nn.Module):
    def __init__(self, widths=(32, 64, 128, 256, 256)):
        super().__init__()
        w = list(widths)
        self.inc = Block(2, w[0])
        self.down = nn.ModuleList()
        self.enc = nn.ModuleList()
        for i in range(1, len(w)):
            self.down.append(nn.Sequential(nn.Conv2d(w[i - 1], w[i], 2, 2, bias=False), nn.BatchNorm2d(w[i]), nn.SiLU()))
            self.enc.append(Block(w[i], w[i]))
        self.mid = Block(w[-1], w[-1])
        self.up = nn.ModuleList()
        self.dec = nn.ModuleList()
        for i in range(len(w) - 1, 0, -1):
            self.up.append(nn.ConvTranspose2d(w[i], w[i - 1], 2, 2))
            self.dec.append(Block(2 * w[i - 1], w[i - 1]))
        self.out = nn.Conv2d(w[0], 1, 1)

    def forward(self, x, m):
        """x: (B, H, W) noisy masked map, m: (H, W) mask. Returns (denoised (B,H,W), bottleneck (B,C,h,w))."""
        h = self.inc(torch.stack([x / SCALE, m.expand_as(x)], 1))
        skips = [h]
        for down, enc in zip(self.down, self.enc):
            h = enc(down(h))
            skips.append(h)
        h = self.mid(h)
        bott = h
        for up, dec, s in zip(self.up, self.dec, reversed(skips[:-1])):
            h = dec(torch.cat([up(h), s], 1))
        y = self.out(h)[:, 0].float() * SCALE
        return y * m, bott


def pool_bottleneck(bott, m):
    """Masked mean and std of every bottleneck channel (per map)."""
    mm = F.adaptive_avg_pool2d(m[None, None], bott.shape[-2:])[0, 0]
    w = mm / mm.sum()
    b = bott.float()
    mean = (b * w).sum((2, 3))
    sd = (((b - mean[:, :, None, None]) ** 2) * w).sum((2, 3)).clamp(min=1e-10).sqrt()
    return torch.cat([mean, sd], 1)
