"""Explicit column-state block encodings W = U_L^dag U_R for the 2D Darcy factor at n = 4
(constant coefficient, base frame: edge and curl blocks on levels n = 2, 4, plus pressure),
built from uniformly controlled Ry cascades (Mottonen), verified by state-vector simulation,
with gate counts and Toffoli-equivalent estimates. 3-point quadrature (H has 256 rows).
"""
import numpy as np, scipy.sparse as sp, json, time
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import UCRYGate
from qiskit.quantum_info import Statevector
from fem2d import hierarchy, prolong_rt0 as prolong_rt, prolong_p1, curl_matrix, chain, rt0_eval
from darcy2d import galerkin

QB3 = np.array([[2 / 3, 1 / 6, 1 / 6], [1 / 6, 2 / 3, 1 / 6], [1 / 6, 1 / 6, 2 / 3]])
QW3 = np.array([1 / 3] * 3)


def build_H():
    ms = hierarchy(4); m = ms[-1]; nt, ne = m.nt, m.ne
    K = np.ones(nt); gamma = 1.0
    rows, cols, vals = [], [], []
    for k in range(3):
        for q in range(3):
            x = np.einsum('k,tkd->td', QB3[q], m.Pv)
            val = rt0_eval(m, np.arange(nt), x, k)
            for c in range(2):
                rows.append((np.arange(nt) * 3 + q) * 2 + c); cols.append(m.TE[:, k]); vals.append(val[:, c])
    R = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(nt * 6, ne))
    wR = np.repeat((m.area[:, None] * QW3[None, :]).ravel(), 2)
    Div = sp.csr_matrix((m.TEs.ravel() / np.repeat(m.area, 3), (np.repeat(np.arange(nt), 3), m.TE.ravel())), shape=(nt, ne))
    bnd = np.zeros(ne, bool)
    for key in m.eb: bnd |= m.eb[key]
    ie = np.where(~bnd)[0]
    MK = (R.T @ sp.diags(wR) @ R)[ie][:, ie]
    W = sp.diags(m.area)
    Pu = (MK + gamma * Div[:, ie].T @ W @ Div[:, ie]).tocsr()
    sw = np.sqrt(m.area / 2)
    Gu = sp.hstack([sp.diags(np.sqrt(wR)) @ R[:, ie], sp.csr_matrix((nt * 6, nt))])
    D = sp.diags(sw) @ Div[:, ie]
    Gs = sp.vstack([Gu, sp.hstack([D, sp.diags(sw)]), sp.hstack([D, -sp.diags(sw)])]).tocsr()
    # frame blocks
    PR = [prolong_rt(ms[0], ms[1])]; PP = [prolong_p1(ms[0], ms[1])]
    Cf = curl_matrix(m)
    blocks, names = [], []
    for l in range(2):
        ml = ms[l]; bl = np.zeros(ml.ne, bool)
        for key in ml.eb: bl |= ml.eb[key]
        Pr = PR[0] if l == 0 else sp.identity(ne, format='csr')
        blocks.append(galerkin(Pr[:, np.where(~bl)[0]][ie].tocsc(), Pu)); names.append('edge n=%d' % ml.n)
    for l in range(2):
        ml = ms[l]; bv = np.zeros(ml.nv, bool)
        for key in ml.vb: bv |= ml.vb[key]
        Pq = PP[0] if l == 0 else sp.identity(m.nv, format='csr')
        blocks.append(galerkin((Cf @ Pq[:, np.where(~bv)[0]])[ie].tocsc(), Pu)); names.append('curl n=%d' % ml.n)
    nu = len(ie)
    Hs = []
    for b in blocks:
        Hs.append((Gs @ sp.vstack([b, sp.csr_matrix((nt, b.shape[1]))])).toarray())
    Hs.append((Gs @ sp.vstack([sp.csr_matrix((nu, nt)), sp.diags(np.sqrt(gamma / m.area))])).toarray())
    names.append('pressure')
    order = [1 - 1, 0, 1, 2, 3, 4]
    return Hs, names


def state_angles(v):
    """Ry-cascade angles (level by level, MSB first) preparing real unit vector v (len 2^n)."""
    n = int(np.log2(len(v)))
    levels = []
    amps = v.copy()
    # norms of subtrees
    for lev in range(n):
        size = 2 ** (n - lev)
        blocks = amps.reshape(-1, size)
        half = size // 2
        left, right = blocks[:, :half], blocks[:, half:]
        if lev < n - 1:
            a, b = np.linalg.norm(left, axis=1), np.linalg.norm(right, axis=1)
        else:
            a, b = left[:, 0], right[:, 0]
        levels.append(2 * np.arctan2(b, a))
    return levels


def colstate_circuit(H):
    """W = U_L^dag U_R acting on (col register c, row register r, flag f).
    Returns circuit, alpha, sizes. Qubit order: c (nc qubits), r (nr), f (1)."""
    R_, m = H.shape
    nr = int(np.ceil(np.log2(R_))); nc = max(1, int(np.ceil(np.log2(m))))
    Hp = np.zeros((2 ** nr, 2 ** nc)); Hp[:R_, :m] = H
    nu = np.linalg.norm(Hp, axis=0).max()
    nz = np.abs(Hp) > 1e-12 * np.abs(Hp).max()
    s = nz.sum(axis=1).max()
    alpha = nu * np.sqrt(s)
    qc = QuantumCircuit(nc + nr + 1)
    cq = list(range(nc)); rq = list(range(nc, nc + nr)); fq = nc + nr
    # U_R: controlled on column j, prepare Hhat[:, j]/||.|| scaled: amplitude sqrt(1-||Hhat_j||^2) to flag
    # Build state on (flag, r) of size 2^(nr+1): first half (flag 0) = Hhat_j, second half (flag 1) = junk on r=0
    targets = [fq] + rq[::-1]            # MSB first: flag, then row bits MSB..LSB
    angles = []
    for j in range(2 ** nc):
        v = np.zeros(2 ** (nr + 1))
        col = Hp[:, j] / nu
        nrm = np.linalg.norm(col)
        if nrm > 0:
            v[:2 ** nr] = col
        v[2 ** nr] = np.sqrt(max(0.0, 1 - nrm ** 2)) if nrm > 0 else 1.0
        angles.append(state_angles(v))
    for lev, t in enumerate(targets):
        ctrl = cq + targets[:lev]
        # UCRY angles ordered by control state integer with ctrl[0] as least significant
        ang = np.zeros(2 ** len(ctrl))
        for j in range(2 ** nc):
            a = angles[j][lev]                              # array over the prefix states of earlier targets
            for pref in range(len(a)):
                # control integer: c bits (j) LSB-first then target prefix bits
                pbits = pref                                # prefix as integer, MSB first over targets[:lev]
                idx = j
                for b in range(lev):
                    bit = (pbits >> (lev - 1 - b)) & 1
                    idx |= bit << (nc + b)
                ang[idx] = a[pref]
        qc.append(UCRYGate(list(ang)), [t] + ctrl)
    UR = qc.copy()
    # U_L: controlled on row i, prepare uniform superposition over N(i) on c, junk to flag
    ql = QuantumCircuit(nc + nr + 1)
    angles = []
    for i in range(2 ** nr):
        v = np.zeros(2 ** (nc + 1))
        Ni = np.where(nz[i])[0]
        if len(Ni):
            v[Ni] = 1 / np.sqrt(s)
        v[2 ** nc] = np.sqrt(max(0.0, 1 - len(Ni) / s))
        angles.append(state_angles(v))
    targets = [fq] + cq[::-1]
    for lev, t in enumerate(targets):
        ctrl = rq + targets[:lev]
        ang = np.zeros(2 ** len(ctrl))
        for i in range(2 ** nr):
            a = angles[i][lev]
            for pref in range(len(a)):
                idx = i
                for b in range(lev):
                    bit = (pref >> (lev - 1 - b)) & 1
                    idx |= bit << (nr + b)
                ang[idx] = a[pref]
        ql.append(UCRYGate(list(ang)), [t] + ctrl)
    qli = QuantumCircuit(nc + nr + 1)
    for inst in reversed(ql.data):
        qli.append(UCRYGate([-a for a in inst.operation.params]), inst.qubits)
    W = UR.compose(qli)
    return W, alpha, nc, nr, s


def _apply_ucry(psi, nq, target, ctrl, angles):
    """psi shaped (2,)*nq; qubit q lives on axis nq-1-q (Qiskit little endian)."""
    ax = lambda q: nq - 1 - q
    order = [ax(target)] + [ax(c) for c in ctrl]
    rest = [a for a in range(nq) if a not in order]
    X = np.transpose(psi, order + rest)
    shp = X.shape
    X = X.reshape(2, 2 ** len(ctrl), -1)
    k = np.arange(2 ** len(ctrl))
    idx = np.zeros_like(k)
    for b in range(len(ctrl)):
        idx |= ((k >> (len(ctrl) - 1 - b)) & 1) << b
    th = np.asarray(angles, float)[idx] / 2
    c, s_ = np.cos(th)[:, None], np.sin(th)[:, None]
    a0, a1 = X[0].copy(), X[1].copy()
    X = np.stack([c * a0 - s_ * a1, s_ * a0 + c * a1]).reshape(shp)
    return np.transpose(X, np.argsort(order + rest))


def simulate(W, psi_vec):
    nq = W.num_qubits
    psi = psi_vec.astype(float).reshape((2,) * nq)
    for inst in W.data:
        op = inst.operation
        assert op.name == 'ucry', op.name
        qs = [W.find_bit(q).index for q in inst.qubits]
        psi = _apply_ucry(psi, nq, qs[0], qs[1:], op.params)
    return psi.reshape(-1)


def verify(W, H, alpha, nc, nr, ntest=3, rng=np.random.default_rng(1)):
    R_, m = H.shape
    errs = []
    for _ in range(ntest):
        c = rng.standard_normal(m); c /= np.linalg.norm(c)
        psi = np.zeros(2 ** (nc + nr + 1)); psi[:m] = c
        out = simulate(W, psi)
        y = np.array([out[i << nc] for i in range(2 ** nr)])[:R_]
        errs.append(np.linalg.norm(alpha * y - H @ c) / np.linalg.norm(H @ c))
    return max(errs)


def gate_model(N, L, b=24, Tcoef=0):
    nb = int(np.ceil(np.log2(1.5 * N)))
    T_idx = (L + 8) * nb; T_qrom = 128 * (L + 1); T_arith = 5 * b * b
    return 2 * (2 * T_idx + T_qrom + T_arith + Tcoef + 2 * b)


if __name__ == '__main__':
    Hs, names = build_H()
    res = []
    for H, nm in list(zip(Hs, names)) + [(np.hstack(Hs), 'full H (direct)')]:
        t = time.time()
        W, alpha, nc, nr, s = colstate_circuit(H)
        err = verify(W, H, alpha, nc, nr)
        tq = transpile(W, basis_gates=['cx', 'rz', 'sx', 'x'], optimization_level=1)
        ops = tq.count_ops()
        nrz = ops.get('rz', 0); ncx = ops.get('cx', 0)
        r = dict(block=nm, shape=list(H.shape), qubits=W.num_qubits, s=int(s), alpha=float(alpha), err=float(err),
                 cx=int(ncx), rz=int(nrz), time=time.time() - t)
        res.append(r); print(json.dumps(r), flush=True)
    json.dump(res, open('circuits_n4.json', 'w'), indent=1)
