"""Print summary tables from whatever result files exist."""
import json, os, collections
import numpy as np


def load(f):
    return [json.loads(l) for l in open(f)] if os.path.exists(f) else []


# ---------------- Biot
R = load('biot_res.jsonl') + load('biot32.jsonl')
if R:
    print('== 2D Biot: worst kappa_alpha over (c0, k~) per lambda; truncated vs untruncated frame')
    for n in sorted(set(r['n'] for r in R)):
        for tr in (True, False):
            rows = [r for r in R if r['n'] == n and r['truncate'] == tr]
            if not rows:
                continue
            byl = collections.defaultdict(list)
            for r in rows:
                byl[r['lam']].append(r['kappa_alpha'])
            w = max(rows, key=lambda r: r['kappa_alpha'])
            print('n=%-3d %-11s gamma=%.4f Gamma=%.3f worst=%.0f (lam=%g)  per lam: %s' % (
                n, 'truncated' if tr else 'untruncated', min(r['gamma'] for r in rows), max(r['Gamma'] for r in rows),
                w['kappa_alpha'], w['lam'], ' '.join('%g:%.0f' % (l, max(v)) for l, v in sorted(byl.items()))))

# ---------------- Case 2
R = load('case2.jsonl')
if R:
    print('\n== 3D Case 2 (geometry not designed for the frame)')
    for r in sorted(R, key=lambda r: (r['n'], r['variant'], r['c'])):
        print('n=%-3d variant=%-4s c=%-7g gamma=%.3g alpha2=%.1f smin=%.3g kappa_alpha=%.3g kappa(A)=%.2g' % (
            r['n'], r['variant'], r['c'], r['gamma'], r['alpha2'], r['smin'], r['kappa_alpha'], r['kappaA']))

# ---------------- Lognormal
for n in (16, 32):
    R = load('lognormal_n%d.jsonl' % n)
    if not R:
        continue
    print('\n== 2D lognormal, n=%d' % n)
    for d in sorted(set(r['decay'] for r in R), reverse=True):
        for s in sorted(set(r['sigma'] for r in R)):
            rr = [r for r in R if r['decay'] == d and r['sigma'] == s]
            if not rr:
                continue
            c = [r['contrast'] for r in rr]
            kb = [r['base']['kappa_alpha'] for r in rr]; kd = [r['decade']['kappa_alpha'] for r in rr]
            print('decay %.1f sigma %g (%d samples): contrast %.1e-%.1e | base ka median %.0f worst %.0f (gamma min %.3f) | decade-mode ka worst %.0f | plain kappa(A) worst %.1e' % (
                d, s, len(rr), min(c), max(c), np.median(kb), max(kb), min(r['base']['gamma'] for r in rr), max(kd), max(r['kappaA'] for r in rr)))

# ---------------- Case 1-M frames
R = load('case1m_frames.jsonl')
if R:
    print('\n== Case 1-M frames')
    for r in sorted(R, key=lambda r: (r['n'], r['J'], not r['LA'], r['cfg'])):
        print('n=%-3d J=%-4d %-6s %-9s alpha2=%5.0f gamma=%s Gamma=%s ||g~|| ||b~||=%.1f smin=%s' % (
            r['n'], r['J'], 'LA' if r['LA'] else 'plain', r['cfg'], r['alpha2'],
            '%.4f' % r['gamma'] if 'gamma' in r else '-', '%.2f' % r['Gamma'] if 'Gamma' in r else '-',
            r['gl'] * r['gb'], '%.4f' % r['smin'] if 'smin' in r else '-'))

# ---------------- circuits
if os.path.exists('circuits_n4.json'):
    print('\n== n = 4 column-state circuits')
    for r in json.load(open('circuits_n4.json')):
        print('%-16s %s qubits=%d s=%d alpha=%.4f err=%.1e CX=%d Rz=%d' % (r['block'], r['shape'], r['qubits'], r['s'], r['alpha'], r['err'], r['cx'], r['rz']))
