"""Iterative extreme eigenvalues for frame bounds and sigma_min(S)."""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def _splu(M):
    return spla.splu(sp.csc_matrix(M), permc_spec='COLAMD')


def frame_bounds_iter(F, P, k=3, tol=1e-8, ncv=60, Plu=None):
    """gamma, Gamma for F as frame of P: eigenvalues of pencil (FF^T, P^{-1})."""
    F = sp.csr_matrix(F); Ft = F.T.tocsr()
    N = F.shape[0]
    Plu = Plu or _splu(P)
    Bop = spla.LinearOperator((N, N), matvec=lambda x: F @ (Ft @ x), dtype=float)
    Mop = spla.LinearOperator((N, N), matvec=lambda x: Plu.solve(x), dtype=float)
    Minv = spla.LinearOperator((N, N), matvec=lambda x: P @ x, dtype=float)
    hi = spla.eigsh(Bop, k=k, M=Mop, Minv=Minv, which='LA', tol=tol, ncv=ncv, return_eigenvectors=False)
    lo = spla.eigsh(Bop, k=k, M=Mop, Minv=Minv, which='SA', tol=tol, ncv=ncv, maxiter=20000, return_eigenvectors=False)
    return float(np.min(lo)), float(np.max(hi)), np.sort(lo)


def binv_pcg(F, P, tol=1e-12, maxiter=5000):
    """solver x -> (FF^T)^{-1} x by CG preconditioned with P (B ~ P^{-1})."""
    F = sp.csr_matrix(F); Ft = F.T.tocsr()
    N = F.shape[0]
    Bop = spla.LinearOperator((N, N), matvec=lambda x: F @ (Ft @ x), dtype=float)
    Pre = spla.LinearOperator((N, N), matvec=lambda x: P @ x, dtype=float)
    its = []

    def solve(r):
        cnt = [0]
        def cb(xk): cnt[0] += 1
        x, info = spla.cg(Bop, r, rtol=tol, maxiter=maxiter, M=Pre, callback=cb)
        its.append(cnt[0])
        if info != 0:
            raise RuntimeError('pcg failed %d' % info)
        return x
    return solve, its


def sigma_min_S(F, A, P, k=4, tol=1e-8):
    """smallest |mu| of A x = mu (FF^T)^{-1} x (nonzero spectrum of S = F^T A F)."""
    N = A.shape[0]
    Alu = _splu(A)
    solveB, its = binv_pcg(F, P)
    Mop = spla.LinearOperator((N, N), matvec=solveB, dtype=float)
    OPinv = spla.LinearOperator((N, N), matvec=lambda x: Alu.solve(x), dtype=float)
    vals = spla.eigsh(A, k=k, M=Mop, sigma=0.0, OPinv=OPinv, which='LM', tol=tol,
                      return_eigenvectors=False)
    return float(np.min(np.abs(vals))), np.sort(vals), its
