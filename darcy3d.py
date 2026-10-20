"""3D mixed Darcy on the unit cube (RT0 x Q0, hexahedra), pressure on x = 0,1, no flow elsewhere.
Shared factor, fitted norm (gamma = 1/K_min), multilevel frame with face and curl blocks,
through-flow columns, lens-avoiding correction (Remark: Proposition LA of the full report).
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from fem3d import Hex, prolong_rt, prolong_ned, chain, G1


def box_mask(cen, box):
    x0, x1, y0, y1, z0, z1 = box
    return ((cen[:, 0] > x0) & (cen[:, 0] < x1) & (cen[:, 1] > y0) & (cen[:, 1] < y1) &
            (cen[:, 2] > z0) & (cen[:, 2] < z1))


LENS1 = (0.25, 0.5, 0.125, 0.375, 0.25, 0.75)
LENS2 = (0.625, 0.875, 0.25, 0.5, 0.125, 0.375)
CHANNEL = (0.0, 1.0, 0.625, 0.75, 0.375, 0.625)


def case1_field(m, klens, kchan):
    cen = m.cell_centers()
    K = np.ones(m.nc)
    for b in (LENS1, LENS2):
        K[box_mask(cen, b)] = klens
    K[box_mask(cen, CHANNEL)] = kchan
    return K


class Darcy3D:
    def __init__(self, m, K):
        self.m, self.K = m, K
        self.R = m.vel_eval()
        w = (m.h ** 3 / 8) / K
        self.wR = np.repeat(w, 24)                     # per (cell, gp, comp), gp-major inside cell
        self.MK = (self.R.T @ sp.diags(self.wR) @ self.R).tocsr()
        self.B = m.div()                               # integral of divergence
        self.W = np.full(m.nc, m.h ** 3)
        self.ff = m.free_faces()
        self.gamma = 1.0 / K.min()

    def system(self):
        ff = self.ff
        MK = self.MK[ff][:, ff]
        B = self.B[:, ff]
        A = sp.bmat([[MK, B.T], [B, None]]).tocsr()
        Pu = (MK + self.gamma * B.T @ sp.diags(1 / self.W) @ B).tocsr()
        Pp = sp.diags(self.W / self.gamma)
        return A, Pu, Pp

    def factor(self):
        m, ff = self.m, self.ff
        nc = m.nc
        tau = np.sqrt(self.gamma)
        Gu = sp.diags(np.sqrt(self.wR)) @ self.R[:, ff]
        Gu = sp.hstack([Gu, sp.csr_matrix((Gu.shape[0], nc))])
        # split rows: a = Div u = B u / |T|, b = p ; weight |T|/2
        sw = np.sqrt(self.W / 2)
        D = sp.diags(sw / self.W) @ self.B[:, ff]
        Ip = sp.diags(sw)
        S1 = sp.hstack([tau * D, Ip / tau]); S2 = sp.hstack([tau * D, -Ip / tau])
        Gs = sp.vstack([Gu, S1, S2]).tocsr()
        J = np.r_[np.ones(Gu.shape[0] + nc), -np.ones(nc)]
        return Gs, J

    def solve_flow(self):
        """velocity fluxes for p = 1 on x = 0, p = 0 on x = 1."""
        m = self.m
        A, _, _ = self.system()
        ff = self.ff
        rhs = np.zeros(A.shape[0])
        x0 = m.xf(0, *np.meshgrid(np.arange(m.n), np.arange(m.n), indexing='ij'))
        pos = {f: i for i, f in enumerate(ff)}
        for f in np.ravel(x0):
            rhs[pos[f]] = 1.0
        sol = spla.spsolve(A.tocsc(), rhs)
        u = np.zeros(m.nf); u[ff] = sol[:len(ff)]
        return u, sol


def galerkin(Gc, P):
    Gc = sp.csc_matrix(Gc)
    d = np.asarray(Gc.multiply(P @ Gc).sum(axis=0)).ravel()
    keep = d > 1e-30 * max(d.max(), 1e-300)
    return Gc[:, np.where(keep)[0]] @ sp.diags(1 / np.sqrt(d[keep]))


def throughflow_columns(ms, Lc_n, lens_boxes, chan_box=CHANNEL, lens_value=1e-12):
    """geometry-only columns on the finest mesh: uniform channel flow and a coarse through-flow
    solve with the lenses impermeable (computed on the n = Lc_n mesh, prolongated)."""
    mf = ms[-1]
    cols = []
    # uniform flow along the channel
    if chan_box is not None:
        n = mf.n
        J, K = np.meshgrid(np.arange(n), np.arange(n), indexing='ij')
        yc, zc = (J.ravel() + .5) / n, (K.ravel() + .5) / n
        inside = (yc > chan_box[2]) & (yc < chan_box[3]) & (zc > chan_box[4]) & (zc < chan_box[5])
        u = np.zeros(mf.nf)
        for i in range(n + 1):
            u[mf.xf(i, J.ravel()[inside], K.ravel()[inside])] = mf.h ** 2
        cols.append(u)
    # coarse through-flow solve
    lev = [m.n for m in ms].index(Lc_n)
    mc = ms[lev]
    cen = mc.cell_centers()
    Kc = np.ones(mc.nc)
    for b in lens_boxes:
        Kc[box_mask(cen, b)] = lens_value
    dz = Darcy3D(mc, Kc)
    dz.gamma = 1.0                       # only the solve is used
    u, _ = dz.solve_flow()
    PR = [prolong_rt(ms[l], ms[l + 1]) for l in range(len(ms) - 1)]
    Pr = chain(PR, lev, len(ms) - 1) if lev < len(ms) - 1 else sp.identity(mf.nf, format='csr')
    cols.append(Pr @ u)
    return np.array(cols).T


def frame3d(ms, Pu, ff, extra_cols=None, lens_correction=None, levels=None):
    """velocity frame blocks on finest free faces, Galerkin scaled with Pu.
    levels: list of level indices into ms to use (default all but the n=1 level)."""
    L = len(ms) - 1
    mf = ms[-1]
    PR = [prolong_rt(ms[l], ms[l + 1]) for l in range(L)]
    PN = [prolong_ned(ms[l], ms[l + 1]) for l in range(L)]
    Cf = mf.curl()
    blocks, names = [], []
    if levels is None:
        levels = list(range(L + 1))
    for l in levels:
        Pr = chain(PR, l, L) if l < L else sp.identity(mf.nf, format='csr')
        g = Pr[:, ms[l].free_faces()][ff]
        blocks.append(galerkin(g, Pu)); names.append(('face', ms[l].n))
    for l in levels:
        fe = ms[l].free_edges()
        if len(fe) == 0:
            continue
        Pn = chain(PN, l, L) if l < L else sp.identity(mf.ned, format='csr')
        g = (Cf @ Pn[:, fe])[ff]
        if lens_correction is not None and ms[l].n in lens_correction['levels']:
            g = lens_correction['C'] @ g
        blocks.append(galerkin(g, Pu)); names.append(('curl', ms[l].n))
    if extra_cols is not None:
        blocks.append(galerkin(sp.csc_matrix(extra_cols[ff]), Pu)); names.append(('through', mf.n))
    return blocks, names


def alpha_blocks(Gs, blocks, nu, npres, Fp):
    alpha2 = 0.0; sks = []
    for Fb in list(blocks) + [None]:
        if Fb is None:
            Fk = sp.vstack([sp.csr_matrix((nu, npres)), Fp])
        else:
            Fk = sp.vstack([Fb, sp.csr_matrix((npres, Fb.shape[1]))])
        Hk = (Gs @ Fk).tocsr()
        Hk.data[np.abs(Hk.data) < 1e-12 * np.abs(Hk.data).max()] = 0
        Hk.eliminate_zeros()
        sr = np.diff(Hk.indptr).max()
        cn2 = np.asarray(Hk.multiply(Hk).sum(axis=0)).ravel().max()
        alpha2 += sr * cn2; sks.append(int(sr))
    return alpha2, sks
