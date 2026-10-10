"""Deliberately invalid: log power in 8 bands minus the BATCH mean (depends on the other maps in the batch)."""
import torch


def features(maps, mask):
    f = torch.fft.rfft2(maps * mask.float())
    p = (f.real ** 2 + f.imag ** 2).reshape(len(maps), 8, -1).mean(2)
    lp = torch.log(p)
    return lp - lp.mean(0, keepdim=True)
