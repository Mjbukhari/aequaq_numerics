"""2D Darcy on lognormal permeability fields (outside Assumption (G)).
log K = sigma * sum_{0<i+j, i,j<5} xi_ij (1+i^2+j^2)^{-1} cos(i pi x) cos(j pi y), xi ~ N(0,1).
Frames: 'base' (edge + curl blocks, Galerkin scaled) and 'decade' (base + one coarse outer mode per
floating connected component of {K < K_max 10^{-j}}, j = 1, 2, ...).
"""
import numpy as np, scipy.sparse as sp, scipy.sparse.csgraph as csg, json, sys, time
from fem2d import hierarchy, prolong_p1, curl_matrix, chain
from darcy2d import Darcy, galerkin, frame_bounds, sigma_S, alpha_mu0, v5_frame


def lognormal_field(m, sigma, xi, decay=1.0):
    cen = m.Pv.mean(axis=1)
    s = np.zeros(m.nt); k = 0
    for i in range(5):
        for j in range(5):
            if i + j == 0:
                continue
            s += xi[k] * np.cos(i * np.pi * cen[:, 0]) * np.cos(j * np.pi * cen[:, 1]) / (1 + i * i + j * j) ** decay
            k += 1
    return np.exp(sigma * s)


def tri_adjacency(m):
    # triangles sharing an edge
    TE = m.TE
    rows = np.repeat(np.arange(m.nt), 3); cols = TE.ravel()
    I = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(m.nt, m.ne))
    return (I @ I.T).tocsr()


def decade_modes(ms, K):
    """list of P1 vertex vectors on the finest mesh (prolongated coarse indicators)."""
    mf = ms[-1]; L = len(ms) - 1
    PP = [prolong_p1(ms[l], ms[l + 1]) for l in range(L)]
    adj = tri_adjacency(mf)
    modes = []
    j = 1
    while True:
        thr = K.max() * 10.0 ** (-j)
        reg = K < thr
        if not reg.any():
            break
        sub = adj[reg][:, reg]
        ncomp, lab = csg.connected_components(sub, directed=False)
        idx = np.where(reg)[0]
        for c in range(ncomp):
            tris = idx[lab == c]
            # coarsest level whose indicator-closure has no boundary vertex
            cen = mf.Pv[tris].mean(axis=1)
            for l in range(L + 1):
                ml = ms[l]
                tl, _ = ml.locate(cen)
                V = np.unique(ml.T[tl].ravel())
                bnd = np.zeros(ml.nv, bool)
                for key in ml.vb:
                    bnd |= ml.vb[key]
                if not bnd[V].any():
                    phi = np.zeros(ml.nv); phi[V] = 1.0
                    Pq = chain(PP, l, L) if l < L else sp.identity(mf.nv, format='csr')
                    modes.append(Pq @ phi)
                    break
        j += 1
    return modes


def run(n, sigma, xi, frames=('base', 'decade'), decay=1.0):
    ms = hierarchy(n)
    mf = ms[-1]
    K = lognormal_field(mf, sigma, xi, decay)
    dz, A, Pu, Pp, blocks, names = v5_frame(ms, K, [], 8, modes=())
    out = dict(n=n, sigma=sigma, decay=decay, contrast=float(K.max() / K.min()))
    Gs, J = dz.factor()
    nt = mf.nt; nu = Pu.shape[0]
    Fp = sp.diags(np.sqrt(dz.gamma / mf.area))
    Cf = curl_matrix(mf)
    for fr in frames:
        bl = list(blocks)
        nmodes = 0
        if fr == 'decade':
            modes = decade_modes(ms, K)
            nmodes = len(modes)
            if modes:
                g = sp.csc_matrix((Cf @ np.array(modes).T)[dz.ie])
                bl.append(galerkin(g, Pu))
        Fu = sp.hstack(bl).tocsr()
        gam, Gam, _ = frame_bounds(Fu, Pu)
        a2 = 0.0
        for Fb in bl:
            sr, cn = alpha_mu0(Gs @ sp.vstack([Fb, sp.csr_matrix((nt, Fb.shape[1]))]))
            a2 += sr * cn ** 2
        sr, cn = alpha_mu0(Gs @ sp.vstack([sp.csr_matrix((nu, nt)), Fp]))
        a2 += sr * cn ** 2
        F = sp.block_diag([Fu, Fp]).tocsr()
        smin, smax = sigma_S(F, A, nker=1)
        out[fr] = dict(gamma=float(gam), Gamma=float(Gam), alpha2=float(a2), smin=float(smin),
                       kappa_alpha=float(a2 / smin), nmodes=nmodes)
    # plain condition number of A (nonzero spectrum)
    ev = np.sort(np.abs(np.linalg.eigvalsh(A.toarray())))
    out['kappaA'] = float(ev[-1] / ev[1])
    return out


if __name__ == '__main__':
    n = int(sys.argv[1]); nsamp = int(sys.argv[2]); decay = float(sys.argv[3])
    sigmas = [float(x) for x in sys.argv[4].split(',')]
    f = open('lognormal_n%d.jsonl' % n, 'a')
    for sigma in sigmas:
        rng = np.random.default_rng(2026)
        for s in range(nsamp):
            xi = rng.standard_normal(24)
            t = time.time()
            r = run(n, sigma, xi, decay=decay); r['sample'] = s; r['time'] = time.time() - t
            f.write(json.dumps(r) + '\n'); f.flush()
            print(decay, sigma, s, 'contrast %.2g base ka %.0f g %.3f decade ka %.0f (modes %d) plain %.2g' %
                  (r['contrast'], r['base']['kappa_alpha'], r['base']['gamma'], r['decade']['kappa_alpha'], r['decade']['nmodes'], r['kappaA']), flush=True)
