"""3D lowest-order Raviart-Thomas (face fluxes) x Q0 and lowest-order Nedelec on uniform
hexahedral meshes of the unit cube, n = 2^L cells per direction, nested by factor 2.

Face flux dofs use the +axis normal. Edge dofs are tangential integrals along +axis.
Boundary: pressure prescribed on x = 0, 1 (natural for mixed form, x-fluxes free);
no flow on y = 0, 1 and z = 0, 1 (those y/z boundary faces removed; Nedelec edges lying
in them removed).
"""
import numpy as np
import scipy.sparse as sp

G1 = np.array([0.5 - 0.5 / np.sqrt(3), 0.5 + 0.5 / np.sqrt(3)])


class Hex:
    def __init__(self, n):
        self.n = n
        self.h = 1.0 / n
        n1 = n + 1
        self.nx_f, self.ny_f, self.nz_f = n1 * n * n, n * n1 * n, n * n * n1
        self.nf = self.nx_f + self.ny_f + self.nz_f
        self.nc = n ** 3
        self.nex, self.ney, self.nez = n * n1 * n1, n1 * n * n1, n1 * n1 * n
        self.ned = self.nex + self.ney + self.nez

    # ---- face indices
    def xf(self, i, j, k):
        n = self.n; return i + (n + 1) * (j + n * k)

    def yf(self, i, j, k):
        n = self.n; return self.nx_f + i + n * (j + (n + 1) * k)

    def zf(self, i, j, k):
        n = self.n; return self.nx_f + self.ny_f + i + n * (j + n * k)

    def cell(self, i, j, k):
        n = self.n; return i + n * (j + n * k)

    # ---- edge indices
    def ex(self, i, j, k):
        n = self.n; return i + n * (j + (n + 1) * k)

    def ey(self, i, j, k):
        n = self.n; return self.nex + i + (n + 1) * (j + n * k)

    def ez(self, i, j, k):
        n = self.n; return self.nex + self.ney + i + (n + 1) * (j + (n + 1) * k)

    def cell_ijk(self):
        n = self.n
        k, j, i = np.meshgrid(np.arange(n), np.arange(n), np.arange(n), indexing='ij')
        return i.ravel(), j.ravel(), k.ravel()       # ordered by cell index

    def cell_centers(self):
        i, j, k = self.cell_ijk()
        return np.c_[(i + .5), (j + .5), (k + .5)] * self.h

    # ---- free sets
    def free_faces(self):
        n = self.n
        mask = np.ones(self.nf, bool)
        # y faces with j in {0,n}, z faces with k in {0,n}
        i, k = np.meshgrid(np.arange(n), np.arange(n), indexing='ij')
        for j in (0, n):
            mask[self.yf(i.ravel(), j, k.ravel())] = False
        i, j = np.meshgrid(np.arange(n), np.arange(n), indexing='ij')
        for kk in (0, n):
            mask[self.zf(i.ravel(), j.ravel(), kk)] = False
        return np.where(mask)[0]

    def free_edges(self):
        n = self.n
        mask = np.ones(self.ned, bool)
        a = np.arange(n + 1); b = np.arange(n)
        # x-edges (i<n, j,k in 0..n): in Gamma_N if j in {0,n} or k in {0,n}
        I, J, K = np.meshgrid(b, a, a, indexing='ij')
        bad = (J == 0) | (J == n) | (K == 0) | (K == n)
        mask[self.ex(I[bad], J[bad], K[bad])] = False
        # y-edges (i in 0..n, j<n, k in 0..n): in Gamma_N if k in {0,n}
        I, J, K = np.meshgrid(a, b, a, indexing='ij')
        bad = (K == 0) | (K == n)
        mask[self.ey(I[bad], J[bad], K[bad])] = False
        # z-edges (i,j in 0..n, k<n): in Gamma_N if j in {0,n}
        I, J, K = np.meshgrid(a, a, b, indexing='ij')
        bad = (J == 0) | (J == n)
        mask[self.ez(I[bad], J[bad], K[bad])] = False
        return np.where(mask)[0]

    # ---- operators
    def div(self):
        """B[cell, face] = +-1 (integral of divergence of unit-flux basis)."""
        i, j, k = self.cell_ijk()
        c = self.cell(i, j, k)
        rows = np.tile(c, 6)
        cols = np.r_[self.xf(i + 1, j, k), self.xf(i, j, k), self.yf(i, j + 1, k), self.yf(i, j, k),
                     self.zf(i, j, k + 1), self.zf(i, j, k)]
        vals = np.r_[np.ones(len(c)), -np.ones(len(c))] .repeat(1)
        vals = np.concatenate([np.ones(len(c)), -np.ones(len(c))] * 3)
        return sp.csr_matrix((vals, (rows, cols)), shape=(self.nc, self.nf))

    def vel_eval(self):
        """rows (cell, gauss point 0..7, comp) -> velocity component values."""
        i, j, k = self.cell_ijk()
        c = self.cell(i, j, k)
        h2 = self.h ** 2
        rows, cols, vals = [], [], []
        q = 0
        for gz in G1:
            for gy in G1:
                for gx in G1:
                    base = (c * 8 + q) * 3
                    # ux
                    rows += [base, base]; cols += [self.xf(i, j, k), self.xf(i + 1, j, k)]
                    vals += [np.full(len(c), (1 - gx) / h2), np.full(len(c), gx / h2)]
                    rows += [base + 1, base + 1]; cols += [self.yf(i, j, k), self.yf(i, j + 1, k)]
                    vals += [np.full(len(c), (1 - gy) / h2), np.full(len(c), gy / h2)]
                    rows += [base + 2, base + 2]; cols += [self.zf(i, j, k), self.zf(i, j, k + 1)]
                    vals += [np.full(len(c), (1 - gz) / h2), np.full(len(c), gz / h2)]
                    q += 1
        rows = np.concatenate([np.broadcast_to(r, len(c)) if np.isscalar(r) else r for r in rows])
        return sp.csr_matrix((np.concatenate(vals), (rows, np.concatenate(cols))), shape=(self.nc * 24, self.nf))

    def curl(self):
        """face flux of curl of edge function = circulation around the face (right-hand rule)."""
        n = self.n
        a, b = np.arange(n + 1), np.arange(n)
        R, C, V = [], [], []
        # x-faces (i in 0..n, j,k < n): +ey(i,j,k) + ez(i,j+1,k) - ey(i,j,k+1) - ez(i,j,k)
        I, J, K = [x.ravel() for x in np.meshgrid(a, b, b, indexing='ij')]
        f = self.xf(I, J, K)
        for e, s in [(self.ey(I, J, K), 1), (self.ez(I, J + 1, K), 1), (self.ey(I, J, K + 1), -1), (self.ez(I, J, K), -1)]:
            R.append(f); C.append(e); V.append(np.full(len(f), float(s)))
        # y-faces (j in 0..n): normal +y, loop z->x: +ez(i,j,k) + ex(i,j,k+1) - ez(i+1,j,k) - ex(i,j,k)
        I, J, K = [x.ravel() for x in np.meshgrid(b, a, b, indexing='ij')]
        f = self.yf(I, J, K)
        for e, s in [(self.ez(I, J, K), 1), (self.ex(I, J, K + 1), 1), (self.ez(I + 1, J, K), -1), (self.ex(I, J, K), -1)]:
            R.append(f); C.append(e); V.append(np.full(len(f), float(s)))
        # z-faces (k in 0..n): normal +z, loop x->y: +ex(i,j,k) + ey(i+1,j,k) - ex(i,j+1,k) - ey(i,j,k)
        I, J, K = [x.ravel() for x in np.meshgrid(b, b, a, indexing='ij')]
        f = self.zf(I, J, K)
        for e, s in [(self.ex(I, J, K), 1), (self.ey(I + 1, J, K), 1), (self.ex(I, J + 1, K), -1), (self.ey(I, J, K), -1)]:
            R.append(f); C.append(e); V.append(np.full(len(f), float(s)))
        return sp.csr_matrix((np.concatenate(V), (np.concatenate(R), np.concatenate(C))), shape=(self.nf, self.ned))


def prolong_rt(mc, mf):
    """coarse face fluxes -> fine face fluxes (n_f = 2 n_c)."""
    nc = mc.n
    R, C, V = [], [], []
    a = np.arange(nc)
    # x-faces of fine mesh: fine i' in 0..2nc, j',k' in 0..2nc-1
    I, J, K = [x.ravel() for x in np.meshgrid(np.arange(2 * nc + 1), np.arange(2 * nc), np.arange(2 * nc), indexing='ij')]
    f = mf.xf(I, J, K); jc, kc = J // 2, K // 2
    even = I % 2 == 0
    R.append(f[even]); C.append(mc.xf(I[even] // 2, jc[even], kc[even])); V.append(np.full(even.sum(), .25))
    od = ~even
    for off in (0, 1):
        R.append(f[od]); C.append(mc.xf(I[od] // 2 + off, jc[od], kc[od])); V.append(np.full(od.sum(), .125))
    # y-faces
    I, J, K = [x.ravel() for x in np.meshgrid(np.arange(2 * nc), np.arange(2 * nc + 1), np.arange(2 * nc), indexing='ij')]
    f = mf.yf(I, J, K); ic, kc = I // 2, K // 2
    even = J % 2 == 0
    R.append(f[even]); C.append(mc.yf(ic[even], J[even] // 2, kc[even])); V.append(np.full(even.sum(), .25))
    od = ~even
    for off in (0, 1):
        R.append(f[od]); C.append(mc.yf(ic[od], J[od] // 2 + off, kc[od])); V.append(np.full(od.sum(), .125))
    # z-faces
    I, J, K = [x.ravel() for x in np.meshgrid(np.arange(2 * nc), np.arange(2 * nc), np.arange(2 * nc + 1), indexing='ij')]
    f = mf.zf(I, J, K); ic, jc = I // 2, J // 2
    even = K % 2 == 0
    R.append(f[even]); C.append(mc.zf(ic[even], jc[even], K[even] // 2)); V.append(np.full(even.sum(), .25))
    od = ~even
    for off in (0, 1):
        R.append(f[od]); C.append(mc.zf(ic[od], jc[od], K[od] // 2 + off)); V.append(np.full(od.sum(), .125))
    return sp.csr_matrix((np.concatenate(V), (np.concatenate(R), np.concatenate(C))), shape=(mf.nf, mc.nf))


def prolong_ned(mc, mf):
    """coarse Nedelec edge dofs -> fine edge dofs. E_x is constant in x, bilinear in (y,z)."""
    nc = mc.n
    R, C, V = [], [], []

    def interp(idx2):
        """fine node index in 0..2nc -> list of (coarse index, weight) for linear interpolation."""
        out = []
        for t in idx2:
            if t % 2 == 0:
                out.append([(t // 2, 1.0)])
            else:
                out.append([(t // 2, .5), (t // 2 + 1, .5)])
        return out
    A2 = np.arange(2 * nc + 1); B2 = np.arange(2 * nc)
    for comp in range(3):
        if comp == 0:
            I, J, K = [x.ravel() for x in np.meshgrid(B2, A2, A2, indexing='ij')]
            fe = mf.ex(I, J, K); along, t1, t2 = I, J, K
            cfun = lambda a_, b_, c_: mc.ex(a_, b_, c_)
        elif comp == 1:
            I, J, K = [x.ravel() for x in np.meshgrid(A2, B2, A2, indexing='ij')]
            fe = mf.ey(I, J, K); along, t1, t2 = J, I, K
            cfun = lambda a_, b_, c_: mc.ey(b_, a_, c_)
        else:
            I, J, K = [x.ravel() for x in np.meshgrid(A2, A2, B2, indexing='ij')]
            fe = mf.ez(I, J, K); along, t1, t2 = K, I, J
            cfun = lambda a_, b_, c_: mc.ez(b_, c_, a_)
        ac = along // 2
        for p1 in (0, 1):
            for p2 in (0, 1):
                ev1 = t1 % 2 == 0; ev2 = t2 % 2 == 0
                # weights along t1
                c1 = np.where(ev1, t1 // 2, t1 // 2 + p1)
                w1 = np.where(ev1, 1.0 if p1 == 0 else 0.0, .5)
                c2 = np.where(ev2, t2 // 2, t2 // 2 + p2)
                w2 = np.where(ev2, 1.0 if p2 == 0 else 0.0, .5)
                w = 0.5 * w1 * w2                    # fine edge is half the coarse edge length
                m = w != 0
                R.append(fe[m]); C.append(cfun(ac[m], c1[m], c2[m])); V.append(w[m])
    return sp.csr_matrix((np.concatenate(V), (np.concatenate(R), np.concatenate(C))), shape=(mf.ned, mc.ned))


def chain(Ps, l, L):
    M = None
    for k in range(l, L):
        M = Ps[k] if M is None else Ps[k] @ M
    return M
