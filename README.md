# Aequa-Q numerics rebuild

Fresh Python implementation of the 2D and 3D computations in the paper, written from the
working record (6 Oct 2026), the supplement and the full report. Everything that goes into
the paper is checked against your published numbers first.

Requirements: Python 3.10+, `numpy`, `scipy`; `qiskit` only for `circuits_n4.py`.

```
pip install numpy scipy qiskit
```

Run every command from this folder. All result files are JSON lines and are appended to,
so a stopped run can be restarted; `biot32.py` and `case1m_uq.py` skip finished cases.

## What has already been run

Result files from my runs are already in this folder (so restarts skip finished work);
`results_so_far/` holds a backup copy and `summary.txt`.

| Check or result | Status |
|---|---|
| 2D Darcy, P4/P16, low/high/mixed, n = 8 to 32 | matches published values to all digits |
| 3D Darcy Case 1, 8^3 and 16^3 | matches (N = 1984, outflow 4.0089203, gamma_F 0.0765 / 0.0949, kappa_alpha 514 / 563) |
| n = 4 circuits | CX counts identical to the supplement |
| Cost model of Sec. VIII | reproduces the Case 1 crossovers (2.1e-11 vs 2.0e-11, 7.4e-9 vs 7.3e-9) |
| Biot, truncated vs untruncated, n = 4, 8, 16 (all 21 cases) | done |
| Biot n = 32 | done for lambda = 1e-6, 1, 4 (both frames); rest in `biot32.py` |
| Case 2 (new 3D geometry), 8^3 and 16^3 | done |
| Lognormal, n = 16, 96 samples | done; n = 32 not run |
| Case 1-M | started only; run below |

`python summarize.py` prints all tables from whatever result files exist.

## Quick validation (minutes)

```
python test2d.py           # 2D identities: div curl = 0, factor = operator, prolongations
python val2d.py            # 2D Darcy vs published table (about 2 min)
python test3d.py           # 3D identities, outflow 4.0089203 at 8^3
python case1_it.py 8       # Case 1 at 8^3: gamma 0.0765, alpha2 39, kappa 510 / 514
```

## Remaining runs, in order of importance

Times are rough, for a 2-core laptop with 8 GB RAM; a faster machine will do better.

1. **Biot n = 32, remaining lambda values** (about 10 min per case, ~3 GB RAM):
   ```
   python biot32.py
   ```
   Needed for the worst case over lambda at n = 32; lambda = 4 (untruncated, 1693) is
   probably the worst already.

2. **Lognormal at n = 32** (about 1 min per sample, 24 samples):
   ```
   python lognormal.py 32 6 0.5 2,3,4
   python lognormal.py 32 6 1.0 4
   ```

3. **Case 1-M frames at 16^3** (1 to 3 min per run, 28 runs; then 8 quick runs):
   ```
   python case1m_frames.py 16 4,16,64,256 full
   python case1m_frames.py 16 4,16,64,256 lam
   ```
   Optional, heavy (30+ min per run): `python case1m_frames.py 32 4,64,256 lowhigh`

4. **Case 1-M classical UQ statistics** (about 5000 solves of 1 to 2 s each, several hours;
   restartable, run it overnight):
   ```
   python case1m_uq.py 64,4,16,256 12
   ```
   For a faster first look, use 4 randomizations: `python case1m_uq.py 64,16 4`.

5. **Case 1-M crossover** (seconds, after steps 3 and 4):
   ```
   python case1m_cross.py
   ```
   Prints the quasi-Monte Carlo rate with an 80% bootstrap band for each J, and the
   crossover accuracies. This answers whether the J = 64 row is real.

6. **Joint contrast and mesh sweep for Case 1 at 32^3** (asked for by the referee; about
   20 to 40 min per point, several GB):
   ```
   python case1_it.py 32 sweep
   ```
   Writes `case1_it.jsonl` (gamma_F, Gamma_F, alpha_H^2, sigma_min and kappa_alpha for c = 1e2, 1e4, 1e6).

7. Optional: **Case 2 at 32^3** (30 to 60 min, several GB): `python case2.py 32 b 1e6`

8. Optional: **circuits** (2 min): `python circuits_n4.py`

When done, send back the `.jsonl` files (or the output of `python summarize.py`).

## Files

| File | Purpose |
|---|---|
| `fem2d.py` | three-direction mesh, P1, P2, RT0, quadrature, prolongations, discrete curl |
| `darcy2d.py` | 2D Darcy factor, fitted norm, V5 frame, dense frame bounds and sigma_min(S) |
| `biot2d.py` | 2D total-pressure Biot: factor, two-regime norm, frames (truncated or not) |
| `biot_sweep.py`, `biot32.py` | Biot sweeps (n = 4 to 16; n = 32) |
| `lognormal.py` | lognormal stress test (base frame and decade-mode frame) |
| `fem3d.py`, `darcy3d.py` | 3D RT0 x Q0, Nedelec curls, factor, frames, through-flow columns |
| `eig3d.py` | iterative frame bounds (Lanczos) and sigma_min(S) (shift-invert) |
| `case1_it.py` | Case 1 checks |
| `case2.py` | Case 2: geometry not designed for the frame |
| `case1m.py`, `case1m_frames.py` | Case 1-M lenses, lens-avoiding correction, frame sweep |
| `case1m_uq.py`, `case1m_cross.py` | classical UQ statistics and crossover |
| `crossover_model.py` | cost models of Sec. VIII, gate model of the working record |
| `circuits_n4.py` | explicit column-state circuits at n = 4, verification, gate counts |
| `summarize.py` | prints all tables |

## Conventions to know

- The "mixed" ordering alternates by column (x index), which is what reproduces your tables.
- alpha_H^2 counts each displacement component of a Biot level as its own block (mu_0 count).
  With this convention kappa_S, gamma_F, Gamma_F, kappa_P and theta match your record exactly,
  but kappa_alpha is 3 to 8% higher than your record's; use one convention throughout.
- Case 1-M lens sites use a fixed random order (seed 7); yours is not recoverable, so
  individual values differ slightly from your tables while the trends are comparable.
- My lens-avoiding blocks are less sparse than yours (alpha_H^2 = 90 vs 71 at J = 4).
