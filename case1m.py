"""Case 1-M: Case 1 box, pressure faces and channel, plus J one-cell lenses on the 16^3 grid
(all three cell indices odd in 1..13, never touching each other, the channel or the pressure faces).
Lens-avoiding correction (Proposition LA of the full report) on the unresolved coarse curl blocks.
"""
import numpy as np, scipy.sparse as sp, scipy.sparse.linalg as spla, scipy.linalg as sla
from fem3d import Hex, prolong_rt, prolong_ned, chain
from darcy3d import Darcy3D, box_mask, galerkin, CHANNEL, alpha_blocks

H16 = 1.0 / 16


def sites(seed=7):
    odd = range(1, 14, 2)
    out = []
    # channel cells on the 16 grid: j in 10..11, k in 6..9 (all i)
    for i in odd:
        for j in odd:
            for k in odd:
                # closed-box contact with channel: j in [9,12] and k in [5,10] means touch
                if 9 <= j <= 12 and 5 <= k <= 10:
                    continue
                out.append((i, j, k))
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(out))
    return [out[o] for o in order]


def lens_boxes(J, seed=7):
    S = sites(seed)[:J]
    return [(i * H16, (i + 1) * H16, j * H16, (j + 1) * H16, k * H16, (k + 1) * H16) for (i, j, k) in S]


def field(m, boxes, lens_vals, kchan):
    cen = m.cell_centers()
    K = np.ones(m.nc)
    for b, v in zip(boxes, np.broadcast_to(lens_vals, (len(boxes),))):
        K[box_mask(cen, b)] = v
    K[box_mask(cen, CHANNEL)] = kchan
    return K


def lens_avoid_matrix(m, boxes, ff):
    """C = I - sum_j W_j on the free faces of mesh m."""
    n, h = m.n, m.h
    N = len(ff)
    pos = -np.ones(m.nf, int); pos[ff] = np.arange(N)
    Bfull = m.div()                        # cell x face
    # unit-coefficient RT0 mass (full)
    d1 = Darcy3D(m, np.ones(m.nc))
    M1 = d1.MK
    cen = m.cell_centers()
    rows, cols, vals = [], [], []
    for b in boxes:
        lens_cells = np.where(box_mask(cen, b))[0]
        x0, x1, y0, y1, z0, z1 = b
        ring_box = (x0 - h, x1 + h, y0 - h, y1 + h, z0 - h, z1 + h)
        patch_cells = np.where(box_mask(cen, ring_box))[0]
        ring_cells = np.setdiff1d(patch_cells, lens_cells)
        Bp = Bfull[patch_cells]
        patch_faces = np.unique(Bp.indices)
        lens_faces = np.unique(Bfull[lens_cells].indices)
        # patch boundary faces: faces of patch cells touched by exactly one patch cell
        cnt = np.asarray(abs(Bfull[patch_cells][:, patch_faces]).sum(axis=0)).ravel()
        bnd_faces = patch_faces[cnt == 1]
        ring_faces = np.setdiff1d(np.setdiff1d(patch_faces, lens_faces), bnd_faces)
        assert np.all(pos[lens_faces] >= 0) and np.all(pos[ring_faces] >= 0)
        Dr = Bfull[ring_cells][:, ring_faces].toarray()
        Dl = Bfull[ring_cells][:, lens_faces].toarray()
        Mr = M1[ring_faces][:, ring_faces].toarray()
        Mi = np.linalg.inv(Mr)
        Sx = Dr @ Mi @ Dr.T
        Sp = np.linalg.pinv(Sx, rcond=1e-12)
        Kmat = -Mi @ Dr.T @ Sp @ Dl          # ring fluxes = Kmat @ v_lens
        # W_j v: lens faces <- v (identity), ring faces <- Kmat v_lens
        lf, rf = pos[lens_faces], pos[ring_faces]
        rows.append(lf); cols.append(lf); vals.append(np.ones(len(lf)))
        R, Cc = np.meshgrid(rf, lf, indexing='ij')
        rows.append(R.ravel()); cols.append(Cc.ravel()); vals.append(Kmat.ravel())
    W = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(N, N))
    return (sp.identity(N, format='csr') - W).tocsr()


def raw_frame(ms, ff, boxes, correct=True, unresolved=(2, 4, 8), through_mesh=16):
    """unscaled velocity frame blocks (list) and names."""
    L = len(ms) - 1
    mf = ms[-1]
    PR = [prolong_rt(ms[l], ms[l + 1]) for l in range(L)]
    PN = [prolong_ned(ms[l], ms[l + 1]) for l in range(L)]
    Cf = mf.curl()
    C = lens_avoid_matrix(mf, boxes, ff) if (correct and boxes) else None
    blocks, names = [], []
    for l in range(L + 1):
        Pr = chain(PR, l, L) if l < L else sp.identity(mf.nf, format='csr')
        blocks.append(sp.csc_matrix(Pr[:, ms[l].free_faces()][ff])); names.append(('face', ms[l].n))
    for l in range(L + 1):
        fe = ms[l].free_edges()
        Pn = chain(PN, l, L) if l < L else sp.identity(mf.ned, format='csr')
        g = (Cf @ Pn[:, fe])[ff]
        if C is not None and ms[l].n in unresolved:
            g = C @ g
        blocks.append(sp.csc_matrix(g)); names.append(('curl', ms[l].n))
    # through-flow columns: channel uniform flow + coarse solve with lenses impermeable
    n = mf.n
    Jg, Kg = np.meshgrid(np.arange(n), np.arange(n), indexing='ij')
    yc, zc = (Jg.ravel() + .5) / n, (Kg.ravel() + .5) / n
    inside = (yc > CHANNEL[2]) & (yc < CHANNEL[3]) & (zc > CHANNEL[4]) & (zc < CHANNEL[5])
    u1 = np.zeros(mf.nf)
    for i in range(n + 1):
        u1[mf.xf(i, Jg.ravel()[inside], Kg.ravel()[inside])] = mf.h ** 2
    lev = [mm.n for mm in ms].index(through_mesh)
    mc = ms[lev]
    Kc = field(mc, boxes, 1e-12, 1.0)
    dz = Darcy3D(mc, Kc)
    u2c, _ = dz.solve_flow()
    Pr = chain(PR, lev, L) if lev < L else sp.identity(mf.nf, format='csr')
    u2 = Pr @ u2c
    blocks.append(sp.csc_matrix(np.c_[u1, u2][ff])); names.append(('through', mf.n))
    return blocks, names


def scaled(blocks, Pu):
    return [galerkin(b, Pu) for b in blocks]
