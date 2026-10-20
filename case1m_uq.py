"""Classical UQ statistics for Case 1-M at 16^3 (channel random variant):
randomized Sobol (R randomizations x 64 points, nested sizes 8..64) and first-order anchored
ANOVA (cut-HDMR) with 3- and 5-point Gauss-Legendre rules. Outflow QoI.
"""
import numpy as np, json, sys, time
from scipy.stats import qmc
from fem3d import Hex
from darcy3d import Darcy3D
from case1m import lens_boxes, field

m = Hex(16)
n = m.n
J0, K0 = np.meshgrid(np.arange(n), np.arange(n), indexing='ij')
OUT_FACES = m.xf(n, J0.ravel(), K0.ravel())


def outflow(J, u):
    boxes = lens_boxes(J)
    lens = 10.0 ** (-3 + 6 * u[:J]); chan = 10.0 ** (1 + 2 * u[J])
    K = field(m, boxes, lens, chan)
    vel, _ = Darcy3D(m, K).solve_flow()
    return float(vel[OUT_FACES].sum())


def run(J, R, out, done):
    d = J + 1
    for r in range(R):
        P = qmc.Sobol(d=d, scramble=True, seed=1000 + r).random(64)
        for i, u in enumerate(P):
            if ('rqmc', J, r, i) in done:
                continue
            t = time.time(); Q = outflow(J, u)
            out.write(json.dumps(dict(kind='rqmc', J=J, r=r, i=i, Q=Q, t=time.time() - t)) + '\n'); out.flush()
    for npt in (3, 5):
        if ('hdmr_done', J, npt) in done:
            continue
        x, w = np.polynomial.legendre.leggauss(npt)
        x = (x + 1) / 2; w = w / 2
        c = np.full(d, 0.5)
        Q0 = outflow(J, c)
        out.write(json.dumps(dict(kind='hdmr', J=J, npt=npt, dim=-1, node=-1, Q=Q0)) + '\n'); out.flush()
        for dim in range(d):
            for k, xi in enumerate(x):
                if abs(xi - 0.5) < 1e-14:
                    continue
                u = c.copy(); u[dim] = xi
                out.write(json.dumps(dict(kind='hdmr', J=J, npt=npt, dim=dim, node=k, Q=outflow(J, u))) + '\n'); out.flush()


if __name__ == '__main__':
    Js = [int(x) for x in sys.argv[1].split(',')]
    R = int(sys.argv[2])
    import os
    done = set()
    if os.path.exists('case1m_uq.jsonl'):
        cnt = {}
        for l in open('case1m_uq.jsonl'):
            u = json.loads(l)
            if u['kind'] == 'rqmc':
                done.add(('rqmc', u['J'], u['r'], u['i']))
            else:
                cnt[(u['J'], u['npt'])] = cnt.get((u['J'], u['npt']), 0) + 1
        for (J, npt), c in cnt.items():
            if c == 1 + (npt - 1) * (J + 1):
                done.add(('hdmr_done', J, npt))
            else:
                print('warning: incomplete HDMR set for J=%d npt=%d; delete those lines and rerun' % (J, npt))
    out = open('case1m_uq.jsonl', 'a')
    for J in Js:
        run(J, R, out, done)
        print('done J', J, flush=True)
