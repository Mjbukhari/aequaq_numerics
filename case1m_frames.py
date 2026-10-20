"""Case 1-M frame sweep: gamma_F, Gamma_F, alpha_H^2 (and sigma_min, Lambda products) vs J."""
import numpy as np, scipy.sparse as sp, json, sys, time
from fem3d import Hex
from darcy3d import Darcy3D, alpha_blocks
from case1m import lens_boxes, field, raw_frame, scaled
from eig3d import frame_bounds_iter, sigma_min_S


def configs(J, seed=11):
    rng = np.random.default_rng(seed + J)
    out = [('low', 1e-6, 1e2), ('high', 1e3, 1e2), ('mid', 1e-3, 10.0)]
    for d in range(3):
        out.append(('random%d' % d, 10 ** rng.uniform(-3, 3, J), 10 ** rng.uniform(1, 3)))
    return out


def lam_products(m, dz, F):
    """||F^T l|| and ||F^T b|| for outflow functional l and pressure-BC rhs b."""
    n = m.n
    ff = dz.ff; nu = len(ff)
    l = np.zeros(m.nf)
    J0, K0 = np.meshgrid(np.arange(n), np.arange(n), indexing='ij')
    for i in range(n + 1):
        l[m.xf(i, J0.ravel(), K0.ravel())] = m.h if 0 < i < n else m.h / 2
    b = np.zeros(m.nf); b[m.xf(0, J0.ravel(), K0.ravel())] = 1.0
    lf = np.r_[l[ff], np.zeros(m.nc)]; bf = np.r_[b[ff], np.zeros(m.nc)]
    return float(np.linalg.norm(F.T @ lf)), float(np.linalg.norm(F.T @ bf))


def run(n, J, correct, cfg, want_gamma=True, want_smin=False):
    t = time.time()
    ms = [Hex(2 ** l) for l in range(1, int(np.log2(n)) + 1)]
    m = ms[-1]
    boxes = lens_boxes(J)
    name, lv, kc = cfg
    K = field(m, boxes, lv, kc)
    dz = Darcy3D(m, K); A, Pu, Pp = dz.system()
    raw, names = raw_frame(ms, dz.ff, boxes, correct=correct)
    bl = scaled(raw, Pu)
    Fu = sp.hstack(bl).tocsr()
    Gs, Jv = dz.factor(); Fp = sp.diags(np.sqrt(dz.gamma / dz.W))
    a2, sk = alpha_blocks(Gs, bl, len(dz.ff), m.nc, Fp)
    out = dict(n=n, J=J, LA=correct, cfg=name, alpha2=a2, sk=sk)
    F = sp.block_diag([Fu, Fp]).tocsr()
    out['gl'], out['gb'] = lam_products(m, dz, F)
    if want_gamma:
        g, G, lo = frame_bounds_iter(Fu, Pu)
        out.update(gamma=g, Gamma=G)
    if want_smin:
        P = sp.block_diag([Pu, Pp]).tocsr()
        out['smin'] = sigma_min_S(F, A, P)[0]
    out['time'] = time.time() - t
    return out


if __name__ == '__main__':
    n = int(sys.argv[1]); Js = [int(x) for x in sys.argv[2].split(',')]
    mode = sys.argv[3]          # 'full' | 'alpha' | 'lowhigh'
    f = open('case1m_frames.jsonl', 'a')
    for J in Js:
        jobs = []
        if mode == 'full':
            jobs = [(False, configs(J)[0])] + [(True, c) for c in configs(J)]
        elif mode == 'lowhigh':
            jobs = [(False, configs(J)[0]), (True, configs(J)[0]), (True, configs(J)[1])]
        elif mode == 'alpha':
            jobs = [(True, configs(J)[0])]
        elif mode == 'lam':
            # worst-case corners for the Lambda product (channel at its maximum 10^3)
            jobs = [(True, ('corner_lo', 1e-3, 1e3)), (True, ('corner_hi', 1e3, 1e3))]
        for corr, cfg in jobs:
            r = run(n, J, corr, cfg, want_gamma=(mode not in ('alpha', 'lam')), want_smin=(mode == 'full' and corr and cfg[0] in ('low', 'random0')))
            r = {k: (float(v) if isinstance(v, np.floating) else v) for k, v in r.items()}
            f.write(json.dumps(r) + '\n'); f.flush()
            print(json.dumps(r), flush=True)
