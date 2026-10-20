import json, time
from darcy2d import run_case
cases = [(8,'P4','high'),(16,'P4','high'),(32,'P4','high'),(8,'P4','low'),(16,'P4','low'),(32,'P4','low'),
         (8,'P4','mixed'),(16,'P4','mixed'),(32,'P4','mixed'),(16,'P16','high'),(16,'P16','low'),(32,'P16','low'),(16,'P16','mixed'),(32,'P16','mixed')]
for n,p,o in cases:
    for ph in ([0,1] if o=='mixed' else [0]):
        t=time.time(); r = run_case(n,p,1e6,o,mixed_phase=ph)
        print(f"{p:4s} {o:6s} ph{ph} n={n:3d} gamma={r['gamma']:.4f} Gamma={r['Gamma']:.3f} smin={r['smin']:.4f} a2={r['alpha2']:.2f} ka={r['kappa_alpha']:.1f}  ({time.time()-t:.0f}s)", flush=True)
