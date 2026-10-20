"""2D total-pressure Biot (P2 x P1 x P1) with the shared factor, two-regime fitted norm
and multilevel frames, following Sec. 7 of the working record of 6 October 2026.

BCs: u = 0 on bottom, left, right (traction-free top); p = 0 on top; pT free.
Unknowns x = [ux, uy, pT, p].
"""
import numpy as np
import scipy.sparse as sp
import scipy.linalg as sla
from fem2d import Mesh, hierarchy, prolong_p1, prolong_p2, p2_grads, chain


def free_p2(m):
    X = m.p2_nodes()
    tol = 1e-12
    bnd = (np.abs(X[:, 0]) < tol) | (np.abs(X[:, 0] - 1) < tol) | (np.abs(X[:, 1]) < tol)
    return np.where(~bnd)[0]


def free_p(m):
    return np.where(~m.vb['top'])[0]


class Biot:
    def __init__(self, m, lam, c0, kt, alpha=1.0):
        self.m, self.lam, self.c0, self.kt, self.al = m, lam, c0, kt, alpha
        nt = m.nt
        self.fu = free_p2(m); self.fp = free_p(m)
        nu1 = len(self.fu)
        self.nu, self.npT, self.npp = 2 * nu1, m.nv, len(self.fp)
        mapu = -np.ones(m.nv + m.ne, int); mapu[self.fu] = np.arange(nu1)
        mapp = -np.ones(m.nv, int); mapp[self.fp] = np.arange(self.npp)
        off_uy, off_pT, off_p = nu1, 2 * nu1, 2 * nu1 + m.nv
        self.N = off_p + self.npp
        tdof = m.p2_tdofs()
        rows, cols, vals, wts, sg = [], [], [], [], []
        r = 0
        lamb = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1.0]])
        from fem2d import QB
        P2G = [None] * 6
        for t in range(nt):
            Gt = m.G[t]
            Tv = m.T[t]
            for q in range(6):
                w = m.qw[t, q]
                l = QB[q]
                g2 = p2_grads(l, Gt)          # 6 x 2
                p1v = l                        # 3
                gx = {}
                # displacement contributions
                ux = [(mapu[tdof[t, a]], g2[a]) for a in range(6) if mapu[tdof[t, a]] >= 0]
                def add_row(entries, weight, sign):
                    nonlocal r
                    for (c, v) in entries:
                        if v != 0.0:
                            rows.append(r); cols.append(c); vals.append(v)
                    wts.append(abs(weight)); sg.append(sign); r += 1
                e11 = [(j, gr[0]) for j, gr in ux]
                e22 = [(off_uy + j, gr[1]) for j, gr in ux]
                e12 = [(j, gr[1] / np.sqrt(2)) for j, gr in ux] + [(off_uy + j, gr[0] / np.sqrt(2)) for j, gr in ux]
                div = [(j, gr[0]) for j, gr in ux] + [(off_uy + j, gr[1]) for j, gr in ux]
                pT = [(off_pT + Tv[a], p1v[a]) for a in range(3)]
                pp = [(off_p + mapp[Tv[a]], p1v[a]) for a in range(3) if mapp[Tv[a]] >= 0]
                add_row(e11, 2 * w, 1); add_row(e22, 2 * w, 1); add_row(e12, 2 * w, 1)
                add_row(div + pT, w / 2, 1)
                add_row(div + [(c, -v) for c, v in pT], w / 2, -1)
                add_row([(c, v / np.sqrt(lam)) for c, v in pT] + [(c, alpha * v / np.sqrt(lam)) for c, v in pp], w, -1)
                if c0 > 0:
                    add_row([(c, np.sqrt(c0) * v) for c, v in pp], w, -1)
                if kt > 0:
                    gp = [(off_p + mapp[Tv[a]], Gt[a]) for a in range(3) if mapp[Tv[a]] >= 0]
                    add_row([(c, np.sqrt(kt) * g[0]) for c, g in gp], w, -1)
                    add_row([(c, np.sqrt(kt) * g[1]) for c, g in gp], w, -1)
        G = sp.csr_matrix((vals, (rows, cols)), shape=(r, self.N))
        self.Gs = sp.diags(np.sqrt(wts)) @ G           # |Sigma|^{1/2} G
        self.J = np.array(sg, float)
        self.A = (self.Gs.T @ sp.diags(self.J) @ self.Gs).tocsr()
        self.PG = (self.Gs.T @ self.Gs).tocsr()
        # blocks for the norm
        iu = np.arange(self.nu); ipT = off_pT + np.arange(m.nv); ip = off_p + np.arange(self.npp)
        self.iu, self.ipT, self.ip = iu, ipT, ip
        # assemble Aeps, MT, Mp, Lp, MTp directly from the factor pieces
        self.Aeps = self._assemble_aeps()
        MT1 = self._p1_mass(m)
        self.MT = MT1
        self.Mp = MT1[self.fp][:, self.fp]
        L1 = self._p1_stiff(m)
        self.Lp = L1[self.fp][:, self.fp]
        self.E = sp.csr_matrix((np.ones(self.npp), (self.fp, np.arange(self.npp))), shape=(m.nv, self.npp))

    def _assemble_aeps(self):
        Gs, J = self.Gs, self.J
        # rows with sign +1 and only u-columns are the strain rows: take the first three rows per qp
        U = Gs[:, self.iu]
        # strain rows: identify rows with no pressure entries and sign +1
        rp = np.asarray(abs(Gs[:, self.nu:]).sum(axis=1)).ravel()
        mask = (rp == 0) & (J > 0)
        Us = U[np.where(mask)[0]]
        return (Us.T @ Us).tocsr()

    @staticmethod
    def _p1_mass(m):
        rows, cols, vals = [], [], []
        for a in range(3):
            for b in range(3):
                rows.append(m.T[:, a]); cols.append(m.T[:, b])
                vals.append(m.area * (2.0 if a == b else 1.0) / 12)
        return sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(m.nv, m.nv))

    @staticmethod
    def _p1_stiff(m):
        rows, cols, vals = [], [], []
        for a in range(3):
            for b in range(3):
                rows.append(m.T[:, a]); cols.append(m.T[:, b])
                vals.append(m.area * np.einsum('td,td->t', m.G[:, a], m.G[:, b]))
        return sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(m.nv, m.nv))

    def regime(self):
        return 'pT' if self.lam > 4 else 's'

    def reaction(self):
        a2 = self.al ** 2
        return self.c0 + (a2 / self.lam if self.lam > 4 else a2 / (1 + self.lam))

    def norm_blocks(self):
        """blocks (Xu, Xmid, Z) of the norm in the regime's variables."""
        lam = self.lam
        if lam > 4:
            Xmid = 0.25 * (1 + 1 / lam) * self.MT
        else:
            Xmid = (1 + 1 / lam) * self.MT
        Z = self.reaction() * self.Mp + self.kt * self.Lp
        return self.Aeps, Xmid.tocsr(), Z.tocsr()

    def Tmat(self):
        """x = T y; identity for lam > 4; T(u,s,p) = (u, s - alpha E p, p) for lam <= 4."""
        N = self.N
        if self.lam > 4:
            return sp.identity(N, format='csr')
        T = sp.lil_matrix((N, N))
        T.setdiag(1.0)
        T = T.tocsr()
        C = sp.csr_matrix((-self.al * np.ones(self.npp), (self.ipT[self.fp], self.ip)), shape=(N, N))
        return (T + C).tocsr()

    def P(self):
        Xu, Xm, Z = self.norm_blocks()
        D = sp.block_diag([Xu, Xm, Z]).tocsr()
        T = self.Tmat()
        Ti = sp.linalg.inv(T.tocsc())
        return (Ti.T @ D @ Ti).tocsr()


def galerkin(Gc, P):
    Gc = sp.csc_matrix(Gc)
    d = np.asarray(Gc.multiply(P @ Gc).sum(axis=0)).ravel()
    keep = d > 1e-300
    return Gc[:, np.where(keep)[0]] @ sp.diags(1 / np.sqrt(d[keep]))


def biot_frame(ms, bi, truncate=True, trunc_factor=4.0):
    """frame blocks in physical variables (list of N x m_k sparse)."""
    L = len(ms) - 1
    mf = ms[-1]
    Xu, Xm, Z = bi.norm_blocks()
    T = bi.Tmat()
    N = bi.N
    nu1 = len(bi.fu)
    P2 = [prolong_p2(ms[l], ms[l + 1]) for l in range(L)]
    P1 = [prolong_p1(ms[l], ms[l + 1]) for l in range(L)]
    blocks, names = [], []
    # displacement BPX, per component, Galerkin-scaled with Aeps
    for l in range(L + 1):
        fl = free_p2(ms[l])
        Pr = chain(P2, l, L) if l < L else sp.identity(mf.nv + mf.ne, format='csr')
        g = Pr[:, fl][bi.fu]                                    # nu1 x n_l
        for comp in range(2):
            Z0 = sp.csr_matrix((nu1, g.shape[1]))
            gu = sp.vstack([g, Z0]) if comp == 0 else sp.vstack([Z0, g])
            gs = galerkin(gu, Xu)
            full = sp.vstack([gs, sp.csr_matrix((N - bi.nu, gs.shape[1]))])
            blocks.append(full); names.append(('u%d' % comp, l))
    # middle block: Jacobi
    dm = Xm.diagonal()
    gm = sp.diags(1 / np.sqrt(dm))
    full = sp.vstack([sp.csr_matrix((bi.nu, mf.nv)), gm, sp.csr_matrix((bi.npp, mf.nv))])
    blocks.append(full); names.append(('mid', L))
    # reaction-diffusion block: P1 BPX on the p space, Galerkin-scaled with Z
    r = bi.reaction()
    hcut = trunc_factor * np.sqrt(bi.kt / r) if r > 0 else np.inf
    for l in range(L + 1):
        hl = ms[l].h
        if truncate and l < L and hl > hcut:
            continue
        fl = free_p(ms[l])
        Pr = chain(P1, l, L) if l < L else sp.identity(mf.nv, format='csr')
        g = Pr[:, fl][bi.fp]
        gs = galerkin(g, Z)
        full = sp.vstack([sp.csr_matrix((bi.nu + mf.nv, gs.shape[1])), gs])
        blocks.append(full); names.append(('p', l))
    # map to physical variables
    blocks = [(T @ b).tocsr() for b in blocks]
    return blocks, names


def analyse(bi, blocks, dense_kappaP=True):
    F = sp.hstack(blocks).tocsr()
    P = bi.P()
    BF = (F @ F.T).toarray()
    Rf = sla.cholesky(BF, lower=True)
    del BF
    Mf = Rf.T @ (P @ Rf)
    evf = sla.eigvalsh(0.5 * (Mf + Mf.T)); del Mf
    Ms = Rf.T @ (bi.A @ Rf)
    evs = np.sort(np.abs(sla.eigvalsh(0.5 * (Ms + Ms.T)))); del Ms, Rf
    alpha2 = 0.0; sks = []
    for b in blocks:
        Hk = (bi.Gs @ b).tocsr()
        Hk.data[np.abs(Hk.data) < 1e-12 * np.abs(Hk.data).max()] = 0
        Hk.eliminate_zeros()
        sr = np.diff(Hk.indptr).max()
        cn2 = np.asarray(Hk.multiply(Hk).sum(axis=0)).ravel().max()
        alpha2 += sr * cn2; sks.append(int(sr))
    out = dict(gamma=evf[0], Gamma=evf[-1], smin=evs[0], smax=evs[-1],
               alpha2=alpha2, kappa_alpha=alpha2 / evs[0], kappaS=evs[-1] / evs[0], sk=sks)
    if dense_kappaP:
        Pd = P.toarray(); Ad = bi.A.toarray()
        Lc = sla.cholesky(Pd, lower=True)
        Li = sla.solve_triangular(Lc, np.eye(len(Pd)), lower=True)
        Ah = Li @ Ad @ Li.T
        ev = np.abs(sla.eigvalsh(0.5 * (Ah + Ah.T)))
        out['kappaP'] = ev.max() / ev.min()
        PG = bi.PG.toarray()
        th = sla.eigvalsh(Li @ PG @ Li.T)
        out['theta'] = th.max()
    return out


def run(n, lam, c0, kt, truncate=True, dense_kappaP=True):
    ms = hierarchy(n)
    bi = Biot(ms[-1], lam, c0, kt)
    blocks, names = biot_frame(ms, bi, truncate=truncate)
    out = analyse(bi, blocks, dense_kappaP=dense_kappaP)
    out.update(n=n, lam=lam, c0=c0, kt=kt, truncate=truncate, N=bi.N,
               nblocks=len(blocks), kept_p_levels=[l for (nm, l) in names if nm == 'p'])
    return out
