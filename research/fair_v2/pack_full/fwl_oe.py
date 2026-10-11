"""Outlier-exposure CNN for inference (copied from agents/outlier/train_oe.py OENet; backbone = fwl_cnn.Net).

Source oe_head (23) = [OoD logit (1), generator-family logits (nfam + 1), parameter predictions (5)], as
agents/outlier/extract_oe.py (first 23 columns of the `oe` source), bf16 autocast.
"""
import torch
import torch.nn as nn

from fwl_cnn import Net


class OENet(nn.Module):
    def __init__(self, nfam, widths=(32, 64, 128, 192, 256, 256)):
        super().__init__()
        self.base = Net(widths)
        self.ood = nn.Sequential(nn.SiLU(), nn.Linear(256, 1))
        self.fam = nn.Sequential(nn.SiLU(), nn.Linear(256, nfam + 1))

    def forward(self, x, m):
        e = self.base.embed(x, m)
        return self.base.head(e), self.ood(e)[:, 0], self.fam(e), e


def oe_head(net, x, m, amp):
    if amp is not None:
        with torch.autocast(x.device.type, dtype=amp):
            pred, logit, flog, _ = net(x, m)
    else:
        pred, logit, flog, _ = net(x, m)
    return torch.cat([logit.float()[:, None], flog.float(), pred.float()], 1)
