"""Case 1-M crossover from the rebuilt frame sweep and classical UQ statistics."""
import json, numpy as np, sys
from crossover_model import T_classical, n_of_eps, quantum_time, crossover

U = [json.loads(l) for l in open('case1m_uq.jsonl')]
FR = [json.loads(l) for l in open('case1m_frames.jsonl')]


def rqmc_stats(J, R=None, rng=None):
    rows = [u for u in U if u['kind'] == 'rqmc' and u['J'] == J]
    Rs = sorted(set(u['r'] for u in rows))
    Q = np.full((len(Rs), 64), np.nan)
    for u in rows:
        Q[Rs.index(u['r']), u['i']] = u['Q']
    Q = Q[~np.isnan(Q).any(axis=1)]
    if rng is not None:                      # bootstrap over randomizations
        Q = Q[rng.integers(0, len(Q), len(Q))]
    mean = Q.mean()
    Ns = [8, 16, 32, 64]
    se = np.array([Q[:, :N].mean(axis=1).std(ddof=1) / mean for N in Ns])
    rate = -np.polyfit(np.log(Ns), np.log(se), 1)[0]
    sd_rel = Q.ravel().std(ddof=1) / mean
    return dict(mean=mean, se64=se[-1], se=se, rate=rate, sd_rel=sd_rel, R=len(Q))


def hdmr(J, npt):
    rows = [u for u in U if u['kind'] == 'hdmr' and u['J'] == J and u['npt'] == npt]
    if not rows:
        return None
    x, w = np.polynomial.legendre.leggauss(npt); x = (x + 1) / 2; w = w / 2
    Q0 = [u['Q'] for u in rows if u['dim'] == -1][0]
    d = J + 1
    est = Q0
    for dim in range(d):
        s = 0.0
        for k in range(npt):
            if abs(x[k] - 0.5) < 1e-14:
                s += w[k] * Q0
            else:
                s += w[k] * [u['Q'] for u in rows if u['dim'] == dim and u['node'] == k][0]
        est += s - Q0
    return est, len(rows)


def alpha_LA(J, n, a16, a32):
    if n <= 16:
        return a16
    if n <= 32:
        return a32
    return a32 + 14 * (np.log2(n) - 5)


def analyse(J, gamma_lb, Lam, a16, a32, st, hd):
    cl_q = lambda e: max(1.0, 64 * (st['se64'] / (e / 2)) ** (1 / st['rate'])) * T_classical(n_of_eps(e))
    def cl_best(e):
        c = cl_q(e)
        for est_n in hd:
            if est_n is None:
                continue
            est, nsolve = est_n
            if abs(est - st['mean']) / st['mean'] <= e / 2:
                c = min(c, nsolve * T_classical(n_of_eps(e)))
        return c
    cl_mc = lambda e: max(1.0, (st['sd_rel'] / (e / 2)) ** 2) * T_classical(n_of_eps(e))
    af = lambda n: alpha_LA(J, n, a16, a32)
    out = {}
    for sc in ('baseline', 'optimistic'):
        q = lambda e: quantum_time(e, af, gamma_lb, Lam, sc)
        e1 = crossover(cl_best, q); e2 = crossover(cl_mc, q)
        out[sc] = dict(eps_best=e1, N_best=4 * n_of_eps(e1) ** 3, t_best=cl_best(e1), eps_mc=e2)
    out['t_classical_1e-3'] = cl_best(1e-3)
    out['t_q_opt_1e-3'] = quantum_time(1e-3, af, gamma_lb, Lam, 'optimistic')
    return out


if __name__ == '__main__':
    Js = sorted(set(u['J'] for u in U))
    # gamma_lb over all LA configurations and J (16^3 and 32^3)
    la = [f for f in FR if f['LA'] and 'gamma' in f]
    gamma_lb = min(f['gamma'] for f in la)
    prods = max(f['gl'] * f['gb'] for f in FR if f['LA'])
    print('gamma_lb = %.4f, max ||g~|| ||b~|| = %.2f' % (gamma_lb, prods))
    rng = np.random.default_rng(0)
    for J in Js:
        st = rqmc_stats(J)
        Lam = prods / (gamma_lb * st['mean'])
        a16 = max(f['alpha2'] for f in FR if f['LA'] and f['J'] == J and f['n'] == 16)
        a32s = [f['alpha2'] for f in FR if f['LA'] and f['J'] == J and f['n'] == 32]
        a32 = max(a32s) if a32s else a16 + 14
        hd = [hdmr(J, 3), hdmr(J, 5)]
        res = analyse(J, gamma_lb, Lam, a16, a32, st, hd)
        # bootstrap the rate
        bs = []
        for b in range(200):
            sb = rqmc_stats(J, rng=rng)
            bs.append(analyse(J, gamma_lb, Lam, a16, a32, sb, hd)['optimistic']['eps_best'])
        lo, hi = np.percentile(bs, [10, 90])
        print('J=%3d R=%d mean=%.4f rate=%.2f se64=%.2e VR=%.0f HDMR3 err=%s HDMR5 err=%s | a16=%.0f a32=%.0f kbar16=%.0f Lam=%.1f' % (
            J, st['R'], st['mean'], st['rate'], st['se64'], (st['sd_rel'] / 8 / st['se64']) ** 2,
            '%.1e' % (abs(hd[0][0] - st['mean']) / st['mean']) if hd[0] else '-',
            '%.1e' % (abs(hd[1][0] - st['mean']) / st['mean']) if hd[1] else '-',
            a16, a32, a16 / gamma_lb, Lam))
        print('       eps* best: baseline %.2e, optimistic %.2e (80%% band %.1e-%.1e) at N=%.1e; vs MC: baseline %.2e, optimistic %.2e; t_cl(1e-3)=%.1f s, t_q,opt(1e-3)=%.2e s' % (
            res['baseline']['eps_best'], res['optimistic']['eps_best'], lo, hi, res['optimistic']['N_best'],
            res['baseline']['eps_mc'], res['optimistic']['eps_mc'], res['t_classical_1e-3'], res['t_q_opt_1e-3']))
