import time, torch
from unet import UNet, denoise
dev='cuda'; torch.backends.cudnn.benchmark=True
m=torch.ones(1424,176,device=dev)
net=UNet().to(dev).eval()
y=torch.randn(32,1424,176,device=dev)*0.03; s=torch.full((32,),0.05,device=dev)
def t(fn, label):
    with torch.no_grad():
        for _ in range(3): fn()
        torch.cuda.synchronize(); t0=time.time()
        for _ in range(6): fn()
        torch.cuda.synchronize(); print(label, f'{(time.time()-t0)/6/32*1000:.2f} ms/map', flush=True)
t(lambda: denoise(net,y,s,m), 'autocast')
a=denoise(net,y,s,m)
nb=UNet().to(dev).eval(); nb.load_state_dict(net.state_dict()); nb=nb.to(torch.bfloat16)
t(lambda: denoise(nb,y,s,m,bf16=True), 'bf16')
nc=torch.compile(nb)
t(lambda: denoise(nc,y,s,m,bf16=True), 'bf16+compile')
