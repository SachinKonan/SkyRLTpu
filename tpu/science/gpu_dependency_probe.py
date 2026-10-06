"""Exercise AbuPlace's compiler dependencies inside the actual GPU sandbox."""
import torch
import triton
import triton.language as tl


@triton.jit
def increment(source, target, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    values = tl.load(source + offsets)
    tl.store(target + offsets, values + 1)


def check():
    import setuptools
    from torch.utils.cpp_extension import load
    source = torch.arange(128, device='cuda', dtype=torch.float32)
    target = torch.empty_like(source)
    increment[(1,)](source, target, BLOCK=128)
    torch.cuda.synchronize()
    assert torch.equal(target, source + 1), 'Triton CUDA preflight returned incorrect output'
    print('GPU dependency preflight passed: setuptools, Torch extension loader, Triton kernel', flush=True)


if __name__ == '__main__':
    check()
