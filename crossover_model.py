"""Cost models of Sec. VIII and crossover search."""
import numpy as np
from scipy.optimize import brentq

T0 = 1.054e-6          # s per N log2 N (single Python core, measured in the full report)
A_RATE, C_ERR = 1.253, 2.29e-2


def T_classical(n):
    N = 4 * n ** 3
    return T0 * N * np.log2(N)


def n_of_eps(eps):
    """mesh with relative bias eps/2 (continuous)."""
    return (2 * C_ERR / eps) ** (1 / A_RATE)


def gate_model(N, L, b=24, Tcoef=0):
    nb = np.ceil(np.log2(1.5 * N))
    return 2 * (2 * (L + 8) * nb + 128 * (L + 1) + 5 * b * b + Tcoef + 2 * b)


def toffoli_per_query(n):
    N = 4 * n ** 3
    return 2 * gate_model(N, np.log2(n))          # one query to S = two calls of U_H


SCEN = {'baseline': dict(k=20, ae=8 * np.pi, cp=0.75, tof=1.0),
        'optimistic': dict(k=1, ae=np.pi, cp=1.0, tof=0.1)}


def quantum_time(eps, alpha_of_n, smin, Lam, scen, t_tof=1e-6):
    s = SCEN[scen]
    n = n_of_eps(eps)
    kb = alpha_of_n(n) / smin
    eq = eps / 2
    M = np.ceil(s['ae'] * Lam / (s['cp'] * eq))
    deg = kb * np.log(2 * kb * Lam / (s['cp'] * eq))
    NU = s['k'] * 2 * M * deg
    return NU * toffoli_per_query(n) * s['tof'] * t_tof


def crossover(classical_time, quantum_time_f, lo=1e-14, hi=1e-1):
    """largest eps at which quantum <= classical (scan log grid then refine)."""
    es = np.logspace(np.log10(hi), np.log10(lo), 2000)
    d = np.array([np.log(quantum_time_f(e)) - np.log(classical_time(e)) for e in es])
    idx = np.where(d <= 0)[0]
    if len(idx) == 0:
        return np.nan
    i = idx[0]
    if i == 0:
        return es[0]
    f = lambda le: np.log(quantum_time_f(10 ** le)) - np.log(classical_time(10 ** le))
    return 10 ** brentq(f, np.log10(es[i]), np.log10(es[i - 1]))


def alpha_case1(n):
    return 39 + 14 * (np.log2(n) - 3)
