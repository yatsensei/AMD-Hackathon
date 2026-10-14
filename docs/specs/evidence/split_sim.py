"""Design-review evidence: StratifiedGroupKFold runtime and what happens to a rare class
that lives in few lots. Synthetic worst case (each class packed into contiguous lots), NOT the
real WM-811K lot structure -- Phase 1 must measure lots-per-class on the real data.

Run: python docs/specs/evidence/split_sim.py
"""
import numpy as np, time
from sklearn.model_selection import StratifiedGroupKFold
rng=np.random.default_rng(0)
counts={"Center":4294,"Donut":555,"Edge-Loc":5189,"Edge-Ring":9680,"Loc":3593,"Near-full":149,"Random":866,"Scratch":1193,"none":147431}
# lots of up to 25 wafers; defects clustered: each lot has a 'dominant' defect drawn by class weight
n=sum(counts.values()); nlots=n//25+1
labels=np.concatenate([[k]*v for k,v in counts.items()])
# cluster: sort defects so same class sits in contiguous lots partially
groups=np.repeat(np.arange(nlots),25)[:n]
# make clustered: put rare classes in few lots
perm=np.concatenate([np.where(labels==c)[0] for c in counts])  # contiguous by class => strong clustering
lab=labels[perm]
t=time.time()
sgkf=StratifiedGroupKFold(n_splits=20,shuffle=True,random_state=42)
fold=np.empty(n,int)
for k,(_,i) in enumerate(sgkf.split(np.zeros(n),lab,groups)): fold[i]=k
print("seconds",round(time.time()-t,1))
split=np.select([fold<3,fold<6],["test","val"],"train")
for c in counts:
    m=lab==c
    print(f"{c:10s} lots={len(np.unique(groups[m])):5d} test={np.sum(m&(split=='test')):6d} val={np.sum(m&(split=='val')):6d}")
