import json, sys, time
from biot2d import run
ns = [int(a) for a in sys.argv[1].split(',')]
lams = [1e-6,1e-2,1,4,16,1e4,1e8]
pairs = [(0,1e-8),(0,1),(1,1e-4)]
out = open(sys.argv[2],'a')
for n in ns:
    for (c0,kt) in pairs:
        for lam in lams:
            for tr in [True, False]:
                t=time.time(); r=run(n,lam,c0,kt,truncate=tr,dense_kappaP=(n<=8))
                r={k:(float(v) if hasattr(v,'dtype') else v) for k,v in r.items()}
                r['sk']=[int(s) for s in r['sk']]; r['time']=time.time()-t
                out.write(json.dumps(r)+'\n'); out.flush()
                print(n,c0,kt,lam,tr,'g=%.4f smin=%.4f ka=%.1f'%(r['gamma'],r['smin'],r['kappa_alpha']), flush=True)
