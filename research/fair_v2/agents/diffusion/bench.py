import sys, time, torch
from unet import UNet, denoise
dev='cuda'; torch.backends.cudnn.benchmark=True
m=torch.ones(1424,176,device=dev)
cfgs={'ps1_32':dict(ch=(32,64,128,192,256),nres=(1,1,2,2,2),ps=1),
      'ps2_64':dict(ch=(64,128,192,256),nres=(1,2,2,2),ps=2),
      'ps2_48':dict(ch=(48,96,160,256),nres=(1,2,2,2),ps=2),
      'ps4_96':dict(ch=(96,160,256),nres=(2,2,2),ps=4)}
for name,cfg in cfgs.items():
  for cl in [False, True]:
    net=UNet(**cfg).to(dev).eval()
    if cl: net=net.to(memory_format=torch.channels_last)
    y=torch.randn(32,1424,176,device=dev)*0.03; s=torch.full((32,),0.05,device=dev)
    with torch.no_grad():
      for _ in range(3): denoise(net,y,s,m)
      torch.cuda.synchronize(); t=time.time()
      for _ in range(5): denoise(net,y,s,m)
      torch.cuda.synchronize(); inf=(time.time()-t)/5/32*1000
    net.train(); opt=torch.optim.AdamW(net.parameters())
    yb=y[:16]; sb=s[:16]
    for i in range(6):
      if i==2: torch.cuda.synchronize(); t=time.time()
      l=denoise(net,yb,sb,m).pow(2).mean(); opt.zero_grad(); l.backward(); opt.step()
    torch.cuda.synchronize(); tr=(time.time()-t)/4*1000
    print(f'{name} cl={cl} params {sum(p.numel() for p in net.parameters())/1e6:.1f}M infer {inf:.2f} ms/map train {tr:.0f} ms/step16', flush=True)
