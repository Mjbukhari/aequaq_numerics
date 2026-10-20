"""2D finite elements on the uniform three-direction mesh of the unit square.

Mesh n x n squares, each split by the diagonal from (i,j) to (i+1,j+1).
Hierarchy n_l = 2 * 2^l (red refinement keeps the three-direction pattern).
Provides P1, P2, RT0 (flux-normalized), P0, quadrature (Dunavant degree 4),
prolongations between nested levels, and the discrete curl P1 -> RT0.
"""
import numpy as np
import scipy.sparse as sp

# ---------------------------------------------------------------- quadrature
_a, _b = 0.445948490915965, 0.091576213509771
QB = np.array([[1 - 2 * _a, _a, _a], [_a, 1 - 2 * _a, _a], [_a, _a, 1 - 2 * _a],
               [1 - 2 * _b, _b, _b], [_b, 1 - 2 * _b, _b], [_b, _b, 1 - 2 * _b]])
QW = np.array([0.223381589678011] * 3 + [0.109951743655322] * 3)
QW = QW / QW.sum()


class Mesh:
    def __init__(self, n):
        self.n = n
        h = 1.0 / n
        self.h = h
        I, J = np.meshgrid(np.arange(n + 1), np.arange(n + 1))
        self.X = np.c_[I.ravel() * h, J.ravel() * h]          # vertex v = j*(n+1)+i
        v = lambda i, j: j * (n + 1) + i
        tris = []
        for j in range(n):
            for i in range(n):
                tris.append([v(i, j), v(i + 1, j), v(i + 1, j + 1)])   # below diagonal
                tris.append([v(i, j), v(i + 1, j + 1), v(i, j + 1)])   # above diagonal
        self.T = np.array(tris)
        nt = len(self.T)
        # edges: local edge k opposite local vertex k: (v_{k+1}, v_{k+2})
        loc = np.stack([self.T[:, [1, 2]], self.T[:, [2, 0]], self.T[:, [0, 1]]], axis=1)  # nt x 3 x 2
        srt = np.sort(loc, axis=2)
        flat = srt.reshape(-1, 2)
        E, inv = np.unique(flat, axis=0, return_inverse=True)
        self.E = E                                    # global orientation a -> b with a < b
        self.TE = inv.reshape(nt, 3)
        self.TEs = np.where(loc[:, :, 0] == srt[:, :, 0], 1.0, -1.0)  # +1 if local dir matches global
        self.nv, self.ne, self.nt = len(self.X), len(E), nt
        P = self.X[self.T]
        d1, d2 = P[:, 1] - P[:, 0], P[:, 2] - P[:, 0]
        self.area = 0.5 * (d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0])
        # barycentric gradients: grad lambda_k
        g = np.zeros((nt, 3, 2))
        for k in range(3):
            a, b = P[:, (k + 1) % 3], P[:, (k + 2) % 3]
            t = b - a
            g[:, k, 0] = t[:, 1] / (2 * self.area)
            g[:, k, 1] = -t[:, 0] / (2 * self.area)
        # sign: grad lambda_k must point toward vertex k
        s = np.einsum('tj,tj->t', g[:, 0], P[:, 0] - P[:, 1])
        g *= np.sign(s)[:, None, None]
        self.G = g
        self.Pv = P
        ex = self.X[E]
        self.emid = ex.mean(axis=1)
        tan = ex[:, 1] - ex[:, 0]
        self.elen = np.linalg.norm(tan, axis=1)
        self.enor = np.c_[tan[:, 1], -tan[:, 0]] / self.elen[:, None]   # right of tangent
        # quadrature points
        self.qx = np.einsum('qk,tkd->tqd', QB, P)                      # nt x 6 x 2
        self.qw = self.area[:, None] * QW[None, :]                     # nt x 6
        # boundary flags
        x, y = self.X[:, 0], self.X[:, 1]
        tol = 1e-12
        self.vb = {'left': np.abs(x) < tol, 'right': np.abs(x - 1) < tol,
                   'bottom': np.abs(y) < tol, 'top': np.abs(y - 1) < tol}
        mx, my = self.emid[:, 0], self.emid[:, 1]
        self.eb = {'left': np.abs(mx) < tol, 'right': np.abs(mx - 1) < tol,
                   'bottom': np.abs(my) < tol, 'top': np.abs(my - 1) < tol}

    # --------------------------------------------------------- locate points
    def locate(self, pts):
        """triangle index and barycentric coordinates for each point (closed triangles)."""
        n = self.n
        x, y = pts[:, 0] * n, pts[:, 1] * n
        i = np.clip(np.floor(x + 1e-12).astype(int), 0, n - 1)
        j = np.clip(np.floor(y + 1e-12).astype(int), 0, n - 1)
        lx, ly = x - i, y - j
        above = ly > lx + 1e-12
        t = 2 * (j * n + i) + above.astype(int)
        P = self.Pv[t]
        # barycentric by solving
        d1, d2 = P[:, 1] - P[:, 0], P[:, 2] - P[:, 0]
        r = pts - P[:, 0]
        det = d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]
        l1 = (r[:, 0] * d2[:, 1] - r[:, 1] * d2[:, 0]) / det
        l2 = (d1[:, 0] * r[:, 1] - d1[:, 1] * r[:, 0]) / det
        lam = np.c_[1 - l1 - l2, l1, l2]
        assert lam.min() > -1e-9, 'point outside located triangle'
        return t, lam

    # --------------------------------------------------------- P2 dofs
    def p2_nodes(self):
        """P2 nodes: vertices then edge midpoints."""
        return np.r_[self.X, self.emid]

    def p2_tdofs(self):
        """local P2 dofs per triangle: 3 vertices then midpoints of local edges 0,1,2
        (edge k opposite vertex k)."""
        return np.c_[self.T, self.nv + self.TE]


# ------------------------------------------------------------- basis values
def p1_vals(lam):
    return lam


def p2_vals(lam):
    l0, l1, l2 = lam[..., 0], lam[..., 1], lam[..., 2]
    return np.stack([l0 * (2 * l0 - 1), l1 * (2 * l1 - 1), l2 * (2 * l2 - 1),
                     4 * l1 * l2, 4 * l2 * l0, 4 * l0 * l1], axis=-1)


def p2_grads(lam, G):
    """lam: (..., 3) barycentric, G: (3,2) barycentric gradients. returns (...,6,2)."""
    l = lam
    out = np.zeros(lam.shape[:-1] + (6, 2))
    for k in range(3):
        out[..., k, :] = (4 * l[..., k] - 1)[..., None] * G[k]
    pairs = [(1, 2), (2, 0), (0, 1)]
    for m, (i, j) in enumerate(pairs):
        out[..., 3 + m, :] = 4 * (l[..., i][..., None] * G[j] + l[..., j][..., None] * G[i])
    return out


# ------------------------------------------------------------- prolongations
def prolong_p1(mc, mf):
    t, lam = mc.locate(mf.X)
    rows = np.repeat(np.arange(mf.nv), 3)
    cols = mc.T[t].ravel()
    vals = lam.ravel()
    Pm = sp.csr_matrix((vals, (rows, cols)), shape=(mf.nv, mc.nv))
    Pm.data[np.abs(Pm.data) < 1e-13] = 0
    Pm.eliminate_zeros()
    return Pm


def prolong_p2(mc, mf):
    pts = mf.p2_nodes()
    t, lam = mc.locate(pts)
    vals = p2_vals(lam)                        # npts x 6
    cols = mc.p2_tdofs()[t]
    rows = np.repeat(np.arange(len(pts)), 6)
    Pm = sp.csr_matrix((vals.ravel(), (rows, cols.ravel())), shape=(len(pts), mc.nv + mc.ne))
    Pm.data[np.abs(Pm.data) < 1e-13] = 0
    Pm.eliminate_zeros()
    return Pm


def rt0_eval(m, t, x, k):
    """value of the global RT0 basis function of local edge k of triangle t at x."""
    xk = m.Pv[t, k]
    return m.TEs[t, k][:, None] * (x - xk) / (2 * m.area[t])[:, None]


def prolong_rt0(mc, mf):
    """fine flux dofs of coarse flux-normalized RT0 functions."""
    # for each fine edge pick one fine triangle containing it, then its coarse parent
    owner = np.zeros(mf.ne, dtype=int)
    owner[mf.TE.ravel()] = np.repeat(np.arange(mf.nt), 3)
    cen = mf.Pv[owner].mean(axis=1)
    tc, _ = mc.locate(cen)
    rows, cols, vals = [], [], []
    x = mf.emid
    for k in range(3):
        val = rt0_eval(mc, tc, x, k)
        flux = np.einsum('ed,ed->e', val, mf.enor) * mf.elen
        rows.append(np.arange(mf.ne)); cols.append(mc.TE[tc, k]); vals.append(flux)
    Pm = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                       shape=(mf.ne, mc.ne))
    Pm.data[np.abs(Pm.data) < 1e-13] = 0
    Pm.eliminate_zeros()
    return Pm


def curl_matrix(m):
    """(C phi_v)_e = phi_v(b) - phi_v(a) for edge a->b: P1 vertex values -> RT0 fluxes."""
    a, b = m.E[:, 0], m.E[:, 1]
    rows = np.r_[np.arange(m.ne), np.arange(m.ne)]
    cols = np.r_[b, a]
    vals = np.r_[np.ones(m.ne), -np.ones(m.ne)]
    return sp.csr_matrix((vals, (rows, cols)), shape=(m.ne, m.nv))


def hierarchy(nfine, ncoarse=2):
    ns = []
    n = ncoarse
    while n <= nfine:
        ns.append(n); n *= 2
    assert ns[-1] == nfine
    return [Mesh(n) for n in ns]


def chain(Ps, l, L):
    """prolongation from level l to level L given list Ps[l] : l -> l+1."""
    M = None
    for k in range(l, L):
        M = Ps[k] if M is None else Ps[k] @ M
    return M
