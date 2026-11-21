# -*- coding: utf-8 -*-
import sys, os, csv
import numpy as np
sys.path.insert(0, 'src')
ROOT = '.'
z = np.load('results/exp23_semantic_embed.npz', allow_pickle=True)
E = z['E']; chars = [str(c) for c in z['chars']]
V = chars + ([] if chr(0xfffd) in chars else [chr(0xfffd)])
Vi = {ch: i for i, ch in enumerate(V)}
M = E.T.copy()
if chr(0xfffd) not in chars:
    r0 = np.random.default_rng(99)
    M = np.hstack([M, r0.standard_normal((M.shape[0], 1))])
    M[:, -1] /= np.linalg.norm(M[:, -1])
rows = list(csv.reader(open('bench/corpus/train__peer-01.csv', encoding='utf-8-sig')))
rows = [r for r in rows[1:] if len(r) >= 3]
msgs = [r[2] for r in rows]
def x_of(s):
    idx = [Vi.get(c, Vi[chr(0xfffd)]) for c in ''.join(ch for ch in str(s) if ch.isprintable())] or [Vi[chr(0xfffd)]]
    v = M[:, idx].sum(1); n = np.linalg.norm(v)
    return v / n if n > 0 else np.zeros(M.shape[0], dtype=np.float32)
X = np.stack([x_of(s) for s in msgs]).astype(np.float32)
n = len(X)
rng = np.random.default_rng(36)
cand = rng.choice(np.arange(3000, n), size=150, replace=False)
mx = []
for i in cand:
    sims = X @ X[i]
    far = np.abs(np.arange(n) - i) > 2000
    mx.append(float(sims[far].max()))
mx = np.array(mx)
print('cross-era max-cos: p10=%.3f p25=%.3f p40=%.3f p50=%.3f p75=%.3f mean=%.3f' % tuple(np.percentile(mx, [10, 25, 40, 50, 75]) ) + ('',) ) if False else print('cross-era max-cos: p10=%.3f p25=%.3f p40=%.3f p50=%.3f p75=%.3f' % tuple(np.percentile(mx, [10, 25, 40, 50, 75])), 'mean=%.3f' % mx.mean())
print('share >=0.5: %.2f  >=0.4: %.2f  >=0.35: %.2f  >=0.3: %.2f' % ((mx >= 0.5).mean(), (mx >= 0.4).mean(), (mx >= 0.35).mean(), (mx >= 0.3).mean()))
