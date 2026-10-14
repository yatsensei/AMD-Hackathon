"""Design-review evidence: model cost, CPU latency, Grad-CAM resolution, scratch survival.

Run: python docs/specs/evidence/bench_models.py   (CPU only; needs torch, torchvision, opencv)
Results quoted in docs/specs/REVIEW_phase1_phase2.md were measured with 4 CPU threads.
"""
import time, torch, torch.nn as nn, torchvision, numpy as np, cv2
torch.set_num_threads(4)
def block(i,o): return nn.Sequential(nn.Conv2d(i,o,3,padding=1,bias=False),nn.BatchNorm2d(o),nn.ReLU(),nn.Conv2d(o,o,3,padding=1,bias=False),nn.BatchNorm2d(o),nn.ReLU(),nn.MaxPool2d(2))
small = nn.Sequential(block(3,32),block(32,64),block(64,128),block(128,256),nn.AdaptiveAvgPool2d(1),nn.Flatten(),nn.Dropout(0.3),nn.Linear(256,9))
def r18(stem):
    m = torchvision.models.resnet18(weights=None, num_classes=9)
    if stem=="s1":
        m.conv1 = nn.Conv2d(3,64,3,1,1,bias=False); m.maxpool = nn.Identity()
    elif stem=="s2":
        m.conv1 = nn.Conv2d(3,64,3,2,1,bias=False); m.maxpool = nn.Identity()
    return m
models = {"small_cnn":small,"resnet18s (3x3 s1, no pool)":r18("s1"),"resnet18 (3x3 s2, no pool)":r18("s2"),"resnet18 imagenet stem":r18("in")}
from torch.utils.flop_counter import FlopCounterMode
for name,m in models.items():
    m.eval()
    x=torch.randn(1,3,64,64)
    with FlopCounterMode(display=False) as fc: m(x)
    gf=fc.get_total_flops()/2e9
    params=sum(p.numel() for p in m.parameters())/1e6
    # last feature map size
    if name=="small_cnn": fm=m[:4](x).shape[-1]
    else:
        f=nn.Sequential(*list(m.children())[:-2]); fm=f(x).shape[-1]
        l3=nn.Sequential(*list(m.children())[:-3])(x).shape[-1]
    res=[]
    with torch.inference_mode():
        for bs in (1,8,40):
            xb=torch.randn(bs,3,64,64)
            for _ in range(5): m(xb)
            t=time.perf_counter(); n=20
            for _ in range(n): m(xb)
            res.append((time.perf_counter()-t)/n*1000)
    print(f"{name:32s} params={params:5.2f}M GMACs={gf:5.3f} lastfeat={fm}x{fm} " + (f"layer3={l3}x{l3} " if name!='small_cnn' else '') + f"ms/call bs1={res[0]:.1f} bs8(TTA)={res[1]:.1f} bs40(5x8 ens+TTA)={res[2]:.1f}")
# scratch survival
H=W=200; m=np.ones((H,W),np.uint8)
yy,xx=np.mgrid[:H,:W]; m[(yy-100)**2+(xx-100)**2>100**2]=0
idx=np.arange(30,170); m[idx,idx]=2
near=cv2.resize(m,(64,64),interpolation=cv2.INTER_NEAREST)
fail=cv2.resize((m==2).astype(np.float32),(64,64),interpolation=cv2.INTER_AREA)
print("scratch dies:",(m==2).sum(),"| nearest fail px:",(near==2).sum(),"| area fail px>0:",(fail>0).sum(), "max frac",fail.max().round(3))
# offset diagonal
m2=m.copy(); m2[m2==2]=1; m2[idx,idx+1]=2
near2=cv2.resize(m2,(64,64),interpolation=cv2.INTER_NEAREST)
print("offset-by-1 scratch | nearest fail px:",(near2==2).sum())
