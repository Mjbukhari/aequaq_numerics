import json, time
from biot2d import run
import os
done=set()
if os.path.exists('biot32.jsonl'):
    for l in open('biot32.jsonl'):
        r=json.loads(l); done.add((r['lam'],r['c0'],r['kt'],r['truncate']))
out=open('biot32.jsonl','a')
cases=[(lam,0,1e-8) for lam in [1e-6,4,1,16,1e-2,1e4,1e8]]+[(1e8,0,1),(4,0,1),(4,1,1e-4)]
for lam,c0,kt in cases:
    for tr in [False, True]:
        if (lam,c0,kt,tr) in done: continue
        t=time.time(); r=run(32,lam,c0,kt,truncate=tr,dense_kappaP=False)
        r={k:(float(v) if hasattr(v,'dtype') else v) for k,v in r.items()}; r['sk']=[int(s) for s in r['sk']]; r['time']=time.time()-t
        out.write(json.dumps(r)+'\n'); out.flush()
        print(lam,c0,kt,tr,'g=%.4f G=%.3f smin=%.4f ka=%.1f t=%.0f'%(r['gamma'],r['Gamma'],r['smin'],r['kappa_alpha'],r['time']),flush=True)
