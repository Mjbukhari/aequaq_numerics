"""2D mixed Darcy (RT0 x P0, u.n = 0) with the shared factor, fitted norm and V5 frame."""
import numpy as np
import scipy.sparse as sp
import scipy.linalg as sla
from fem2d import Mesh, hierarchy, prolong_p1, prolong_rt0, curl_matrix, chain, rt0_eval


def patterns(name, c, order='low', mixed_phase=0):
    """returns list of (xmin,xmax,ymin,ymax,value), ell_c mesh size."""
    if name == 'P4':
        lo = [1 / 8, 5 / 8]; side = 1 / 4; nlc = 8
    elif name == 'P16':
        lo = [1 / 16 + k / 4 for k in range(4)]; side = 1 / 8; nlc = 16
    else:
        raise ValueError
    incl = []
    for iy, y0 in enumerate(lo):
        for ix, x0 in enumerate(lo):
            if order == 'low':
                v = 1.0 / c
            elif order == 'high':
                v = c
            else:
                v = c ** (-0.5) if (ix + mixed_phase) % 2 == 0 else c ** 0.5   # column stripes (matches published)
            incl.append((x0, x0 + side, y0, y0 + side, v))
    return incl, nlc


def perm_field(m, incl):
    cen = m.Pv.mean(axis=1)
    K = np.ones(m.nt)
    lab = np.zeros(m.nt, dtype=int)
    for r, (x0, x1, y0, y1, v) in enumerate(incl):
        ins = (cen[:, 0] > x0) & (cen[:, 0] < x1) & (cen[:, 1] > y0) & (cen[:, 1] < y1)
        K[ins] = v
        lab[ins] = r + 1
    return K, lab


class Darcy:
    def __init__(self, m, K):
        self.m, self.K = m, K
        nt, ne = m.nt, m.ne
        # velocity evaluation at quadrature points: rows (t, q, comp)
        rows, cols, vals = [], [], []
        for k in range(3):
            for q in range(6):
                val = rt0_eval(m, np.arange(nt), m.qx[:, q], k)   # nt x 2
                for c in range(2):
                    rows.append((np.arange(nt) * 6 + q) * 2 + c)
                    cols.append(m.TE[:, k]); vals.append(val[:, c])
        self.R = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                               shape=(nt * 12, ne))
        self.wR = np.repeat((m.qw / K[:, None]).ravel(), 2)
        self.MK = (self.R.T @ sp.diags(self.wR) @ self.R).tocsr()
        self.Div = sp.csr_matrix((m.TEs.ravel() / np.repeat(m.area, 3),
                                  (np.repeat(np.arange(nt), 3), m.TE.ravel())), shape=(nt, ne))
        self.W = sp.diags(m.area)
        self.B = (self.W @ self.Div).tocsr()
        # interior edges (u.n = 0)
        bnd = np.zeros(ne, bool)
        for key in m.eb:
            bnd |= m.eb[key]
        self.ie = np.where(~bnd)[0]
        self.gamma = 1.0 / K.min()

    def system(self):
        ie = self.ie
        MK = self.MK[ie][:, ie]
        B = self.B[:, ie]
        A = sp.bmat([[MK, B.T], [B, None]]).tocsr()
        Pu = (MK + self.gamma * self.Div[:, ie].T @ self.W @ self.Div[:, ie]).tocsr()
        Pp = self.W / self.gamma
        return A, Pu, sp.diags(Pp.diagonal())

    def factor(self):
        """H-rows map: returns (Gs, J) with Gs = |Sigma|^{1/2} G (rows x [u_ie, p]) and signs J."""
        m, ie = self.m, self.ie
        nt = m.nt
        tau = np.sqrt(self.gamma)
        Gu = sp.diags(np.sqrt(self.wR)) @ self.R[:, ie]
        Gu = sp.hstack([Gu, sp.csr_matrix((Gu.shape[0], nt))])
        sw = np.sqrt(m.area / 2)
        D = sp.diags(sw) @ self.Div[:, ie]
        Ip = sp.diags(sw)
        S1 = sp.hstack([tau * D, Ip / tau])
        S2 = sp.hstack([tau * D, -Ip / tau])
        Gs = sp.vstack([Gu, S1, S2]).tocsr()
        J = np.r_[np.ones(Gu.shape[0] + nt), -np.ones(nt)]
        return Gs, J


def galerkin(Gcols, P):
    d = np.asarray((Gcols.multiply(P @ Gcols)).sum(axis=0)).ravel()
    return Gcols @ sp.diags(1 / np.sqrt(d))


def v5_frame(ms, K, incl, nlc, modes=('out', 'in')):
    """velocity frame blocks on finest interior edges, Galerkin-scaled with Pu."""
    L = len(ms) - 1
    mf = ms[-1]
    dz = Darcy(mf, K)
    A, Pu, Pp = dz.system()
    ie = dz.ie
    PR = [prolong_rt0(ms[l], ms[l + 1]) for l in range(L)]
    PP = [prolong_p1(ms[l], ms[l + 1]) for l in range(L)]
    Cf = curl_matrix(mf)
    blocks, names = [], []
    for l in range(L + 1):
        ml = ms[l]
        bl = np.zeros(ml.ne, bool)
        for key in ml.eb:
            bl |= ml.eb[key]
        iel = np.where(~bl)[0]
        Pr = chain(PR, l, L) if l < L else sp.identity(mf.ne, format='csr')
        g = Pr[:, iel][ie]
        blocks.append(galerkin(g.tocsc(), Pu)); names.append(('edge', l))
    for l in range(L + 1):
        ml = ms[l]
        bv = np.zeros(ml.nv, bool)
        for key in ml.vb:
            bv |= ml.vb[key]
        iv = np.where(~bv)[0]
        if len(iv) == 0:
            continue
        Pq = chain(PP, l, L) if l < L else sp.identity(mf.nv, format='csr')
        g = (Cf @ Pq[:, iv])[ie]
        blocks.append(galerkin(g.tocsc(), Pu)); names.append(('curl', l))
    # coarse modes for floating inclusions
    lc = int(np.log2(nlc // 2))
    if 'out' in modes:
        cols = []
        mlc = ms[min(lc, L)]
        Pq = chain(PP, min(lc, L), L) if lc < L else sp.identity(mf.nv, format='csr')
        for (x0, x1, y0, y1, v) in incl:
            X = mlc.X
            phi = ((X[:, 0] > x0 - 1e-12) & (X[:, 0] < x1 + 1e-12) &
                   (X[:, 1] > y0 - 1e-12) & (X[:, 1] < y1 + 1e-12)).astype(float)
            cols.append(Cf @ (Pq @ phi))
        g = sp.csc_matrix(np.array(cols).T)[ie]
        blocks.append(galerkin(g, Pu)); names.append(('out', lc))
    if 'in' in modes:
        cols = []
        li = min(lc + 2, L)
        mli = ms[li]
        Pq = chain(PP, li, L) if li < L else sp.identity(mf.nv, format='csr')
        for (x0, x1, y0, y1, v) in incl:
            X = mli.X
            inside = (X[:, 0] > x0 + 1e-12) & (X[:, 0] < x1 - 1e-12) & (X[:, 1] > y0 + 1e-12) & (X[:, 1] < y1 - 1e-12)
            idx = np.where(inside)[0]
            c = np.array([(x0 + x1) / 2, (y0 + y1) / 2])
            j = idx[np.argmin(np.linalg.norm(X[idx] - c, axis=1))]
            phi = np.zeros(mli.nv); phi[j] = 1.0
            cols.append(Cf @ (Pq @ phi))
        g = sp.csc_matrix(np.array(cols).T)[ie]
        blocks.append(galerkin(g, Pu)); names.append(('in', li))
    return dz, A, Pu, Pp, blocks, names


def frame_bounds(F, P):
    """gamma, Gamma of F as a frame for P (dense)."""
    BF = (F @ F.T).toarray()
    Rf = sla.cholesky(BF, lower=True)
    Pd = P.toarray() if sp.issparse(P) else P
    M = Rf.T @ Pd @ Rf
    ev = sla.eigvalsh(M)
    return ev[0], ev[-1], Rf


def sigma_S(F, A, nker=0):
    BF = (F @ F.T).toarray()
    Rf = sla.cholesky(BF, lower=True)
    M = Rf.T @ (A @ Rf)
    M = 0.5 * (M + M.T)
    ev = np.sort(np.abs(sla.eigvalsh(M)))
    return ev[nker], ev[-1]


def alpha_mu0(Hk, tol=1e-12):
    Hk = sp.csr_matrix(Hk)
    Hk.data[np.abs(Hk.data) < tol * max(1e-300, np.abs(Hk.data).max())] = 0
    Hk.eliminate_zeros()
    sr = np.diff(Hk.indptr).max()
    cn = np.sqrt(np.asarray(Hk.multiply(Hk).sum(axis=0)).ravel()).max()
    return sr, cn


def run_case(nfine, pat, c, order, mixed_phase=0, modes=('out', 'in'), compute_S=True):
    ms = hierarchy(nfine)
    incl, nlc = patterns(pat, c, order, mixed_phase)
    K, lab = perm_field(ms[-1], incl)
    dz, A, Pu, Pp, blocks, names = v5_frame(ms, K, incl, nlc, modes)
    Fu = sp.hstack(blocks).tocsr()
    gam, Gam, _ = frame_bounds(Fu, Pu)
    out = dict(n=nfine, pat=pat, c=c, order=order, gamma=gam, Gamma=Gam)
    # full frame and encoding
    nt = ms[-1].nt
    Fp = sp.diags(np.sqrt(dz.gamma / ms[-1].area))
    Gs, J = dz.factor()
    nu = Pu.shape[0]
    alpha2 = 0.0
    sks = []
    for Fb in blocks:
        Fk = sp.vstack([Fb, sp.csr_matrix((nt, Fb.shape[1]))])
        sr, cn = alpha_mu0(Gs @ Fk)
        alpha2 += sr * cn ** 2; sks.append(sr)
    Fk = sp.vstack([sp.csr_matrix((nu, nt)), Fp])
    sr, cn = alpha_mu0(Gs @ Fk)
    alpha2 += sr * cn ** 2; sks.append(sr)
    out['alpha2'] = alpha2; out['sk'] = sks
    if compute_S:
        F = sp.block_diag([Fu, Fp]).tocsr()
        smin, smax = sigma_S(F, A, nker=1)
        out['smin'] = smin; out['kappa_alpha'] = alpha2 / smin; out['kappaS'] = smax / smin
    return out


if __name__ == '__main__':
    import sys, json, time
    t0 = time.time()
    r = run_case(8, 'P4', 1e6, 'high')
    print(r, time.time() - t0)
