import numpy as np, scipy.sparse as sp, time, sys, json
from fem3d import Hex
from darcy3d import *
from eig3d import frame_bounds_iter, sigma_min_S
def build(n, klens, kchan, through=True, lens_boxes=(LENS1,LENS2), field=None):
    ms=[Hex(2**l) for l in range(1,int(np.log2(n))+1)]
    m=ms[-1]; K=case1_field(m,klens,kchan) if field is None else field(m)
    dz=Darcy3D(m,K); A,Pu,Pp=dz.system()
    extra = throughflow_columns(ms, 8, list(lens_boxes)) if through else None
    blocks,names=frame3d(ms,Pu,dz.ff,extra_cols=extra)
    return ms,dz,A,Pu,Pp,blocks,names
def run(n, klens, kchan, through=True, S=True):
    t=time.time()
    ms,dz,A,Pu,Pp,blocks,names=build(n,klens,kchan,through)
    Fu=sp.hstack(blocks).tocsr()
    g,G,lo=frame_bounds_iter(Fu,Pu)
    Gs,J=dz.factor(); Fp=sp.diags(np.sqrt(dz.gamma/dz.W))
    a2,sks=alpha_blocks(Gs,blocks,len(dz.ff),ms[-1].nc,Fp)
    out=dict(n=n,klens=klens,kchan=kchan,through=through,gamma=g,Gamma=G,low3=list(lo),alpha2=a2,sk=sks)
    if S:
        F=sp.block_diag([Fu,Fp]).tocsr(); P=sp.block_diag([Pu,Pp]).tocsr()
        smin,vals,its=sigma_min_S(F,A,P)
        out.update(smin=smin,kappa_alpha=a2/smin,pcg_its=int(np.mean(its)))
    out['time']=time.time()-t
    return out
if __name__=='__main__':
    n=int(sys.argv[1])
    # default: the comparison configuration and permeability ratio 1e12;
    # 'sweep': joint contrast sweep at this mesh, lenses 1/c and channel c, c = 1e2, 1e4, 1e6
    cfgs=[(1e-4,1e2),(1e-6,1e6)]
    if len(sys.argv)>2 and sys.argv[2]=='sweep':
        cfgs=[(1/c,c) for c in (1e2,1e4,1e6)]
    f=open('case1_it.jsonl','a')
    for kl,kc in cfgs:
        r=run(n,kl,kc); r={k:(float(v) if isinstance(v,(np.floating,)) else v) for k,v in r.items()}
        f.write(json.dumps(r)+'\n'); f.flush(); print(json.dumps(r),flush=True)
