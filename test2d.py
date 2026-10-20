import numpy as np, scipy.sparse as sp
from fem2d import *
from darcy2d import *
ms = hierarchy(16)
m = ms[-1]
C = curl_matrix(m)
dz = Darcy(m, np.ones(m.nt))
print('DivC', abs(dz.Div @ C).max())
# prolongation identity: coarse mass = P^T M_f P
for l in range(len(ms)-1):
    dc, df = Darcy(ms[l], np.ones(ms[l].nt)), Darcy(ms[l+1], np.ones(ms[l+1].nt))
    Pr = prolong_rt0(ms[l], ms[l+1])
    print('RT0 mass', abs(Pr.T @ df.MK @ Pr - dc.MK).max(), 'div', abs((df.B @ Pr).sum() ) )
    Pp = prolong_p1(ms[l], ms[l+1])
    print('curl commute', abs(curl_matrix(ms[l+1]) @ Pp - Pr @ curl_matrix(ms[l])).max())
K = np.exp(np.random.randn(m.nt)); dz = Darcy(m, K)
A, Pu, Pp = dz.system(); Gs, J = dz.factor()
print('GSG=A', abs(Gs.T @ sp.diags(J) @ Gs - A).max(), 'G|S|G=P', abs(Gs.T@Gs - sp.block_diag([Pu,Pp])).max())
