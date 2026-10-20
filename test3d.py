import numpy as np, scipy.sparse as sp
from fem3d import *
from darcy3d import *
ms=[Hex(n) for n in (2,4,8)]
for m in ms:
    C=m.curl(); B=m.div()
    print(m.n, 'div curl', abs(B@C).max(), 'nf',m.nf,'free',len(m.free_faces()),'ned',m.ned,'free e',len(m.free_edges()))
    # curl of free edges has zero flux on non-free faces
    fe=m.free_edges(); ff=m.free_faces(); nonfree=np.setdiff1d(np.arange(m.nf),ff)
    print('   curl leak to no-flow faces', abs(C[nonfree][:,fe]).max() if len(nonfree) else 0)
for l in range(2):
    mc,mf=ms[l],ms[l+1]
    Pr=prolong_rt(mc,mf); Pn=prolong_ned(mc,mf)
    print('commute', abs(mf.curl()@Pn - Pr@mc.curl()).max())
    dc=Darcy3D(mc,np.ones(mc.nc)); df=Darcy3D(mf,np.ones(mf.nc))
    print('mass', abs(Pr.T@df.MK@Pr - dc.MK).max())
    # coarse divergence preserved: fine div of prolongated = restriction-sum
m=ms[-1]; K=case1_field(m,1e-4,1e2); dz=Darcy3D(m,K); A,Pu,Pp=dz.system(); Gs,J=dz.factor()
print('N', A.shape, 'GSG=A', abs(Gs.T@sp.diags(J)@Gs - A).max()/abs(A).max(), 'G|S|G=P', abs(Gs.T@Gs-sp.block_diag([Pu,Pp])).max()/abs(Pu).max())
u,_=dz.solve_flow(); print('outflow', u[m.xf(m.n,*np.meshgrid(np.arange(m.n),np.arange(m.n),indexing='ij'))].sum())
K1=np.ones(m.nc); u1,_=Darcy3D(m,K1).solve_flow(); print('outflow K=1', u1[m.xf(m.n,*np.meshgrid(np.arange(m.n),np.arange(m.n),indexing='ij'))].sum())
