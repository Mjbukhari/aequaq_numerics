"""Case 2: a 3D geometry the frame was not designed for.
L-shaped high-permeability fracture path (inflow slab that dead-ends, joined to a perpendicular
slab that reaches the outflow face), a low-permeability barrier wall touching the boundary,
and one floating lens. All boxes aligned with the 8^3 grid.
"""
import numpy as np, scipy.sparse as sp, scipy.sparse.linalg as spla, time, sys, json
from fem3d import Hex, prolong_rt, chain
from darcy3d import Darcy3D, box_mask, galerkin, frame3d, alpha_blocks
from eig3d import frame_bounds_iter, sigma_min_S

F1 = (0.0, 0.625, 0.125, 0.25, 0.125, 0.875)      # from inflow face, dead-ends at x = 5/8
F2 = (0.5, 1.0, 0.125, 0.875, 0.5, 0.625)         # to outflow face; overlaps F1
BAR = (0.25, 0.375, 0.375, 1.0, 0.0, 1.0)          # wall touching y = 1, z = 0, z = 1
LENS = (0.625, 0.875, 0.625, 0.875, 0.125, 0.375)  # floating
HI = [F1, F2]
LO = [BAR, LENS]


def field(m, c):
    cen = m.cell_centers()
    K = np.ones(m.nc)
    for b in LO:
        K[box_mask(cen, b)] = 1.0 / c
    for b in HI:
        K[box_mask(cen, b)] = c
    return K


def coarse_flow_column(ms, nc_mesh, lo_val, hi_val):
    lev = [m.n for m in ms].index(nc_mesh)
    mc = ms[lev]
    cen = mc.cell_centers()
    Kc = np.ones(mc.nc)
    for b in LO:
        Kc[box_mask(cen, b)] = lo_val
    for b in HI:
        Kc[box_mask(cen, b)] = hi_val
    dz = Darcy3D(mc, Kc)
    u, _ = dz.solve_flow()
    L = len(ms) - 1
    PR = [prolong_rt(ms[l], ms[l + 1]) for l in range(L)]
    Pr = chain(PR, lev, L) if lev < L else sp.identity(ms[-1].nf, format='csr')
    return Pr @ u


def run(n, c, variant, S=True):
    t = time.time()
    ms = [Hex(2 ** l) for l in range(1, int(np.log2(n)) + 1)]
    m = ms[-1]
    K = field(m, c)
    dz = Darcy3D(m, K)
    A, Pu, Pp = dz.system()
    cols = []
    if variant in ('a', 'b'):
        cols.append(coarse_flow_column(ms, 8, 1e-12, 1.0))      # generic: low regions impermeable
    if variant == 'b':
        cols.append(coarse_flow_column(ms, 8, 1e-12, 1e6))      # label-based: high regions open
    extra = np.array(cols).T if cols else None
    blocks, names = frame3d(ms, Pu, dz.ff, extra_cols=extra)
    Fu = sp.hstack(blocks).tocsr()
    g, G, lo = frame_bounds_iter(Fu, Pu)
    Gs, J = dz.factor(); Fp = sp.diags(np.sqrt(dz.gamma / dz.W))
    a2, sks = alpha_blocks(Gs, blocks, len(dz.ff), m.nc, Fp)
    out = dict(case='C2', n=n, c=c, variant=variant, gamma=g, Gamma=G, alpha2=a2, sk=sks)
    if S:
        F = sp.block_diag([Fu, Fp]).tocsr(); P = sp.block_diag([Pu, Pp]).tocsr()
        smin, vals, its = sigma_min_S(F, A, P)
        out.update(smin=smin, kappa_alpha=a2 / smin)
        # plain condition number of A (largest/smallest |eig|) for reference
        Alu = spla.splu(sp.csc_matrix(A))
        N = A.shape[0]
        lmax = spla.eigsh(A, k=1, which='LM', return_eigenvectors=False)[0]
        lmin = spla.eigsh(A, k=1, sigma=0, OPinv=spla.LinearOperator((N, N), matvec=Alu.solve), which='LM', return_eigenvectors=False)[0]
        out['kappaA'] = abs(lmax) / abs(lmin)
    out['time'] = time.time() - t
    return out


if __name__ == '__main__':
    n = int(sys.argv[1])
    variants = sys.argv[2].split(',')
    cs = [float(x) for x in sys.argv[3].split(',')]
    f = open('case2.jsonl', 'a')
    for c in cs:
        for v in variants:
            r = run(n, c, v)
            r = {k: (float(x) if isinstance(x, np.floating) else x) for k, x in r.items()}
            f.write(json.dumps(r) + '\n'); f.flush()
            print(json.dumps(r), flush=True)
