# -*- coding: utf-8 -*-
import sys, csv
import numpy as np
sys.path.insert(0, 'src')
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
n = len(msgs)
def x_of(s):
    idx = [Vi.get(c, Vi[chr(0xfffd)]) for c in ''.join(ch for ch in str(s) if ch.isprintable())] or [Vi[chr(0xfffd)]]
    v = M[:, idx].sum(1); nn = np.linalg.norm(v)
    return v / nn if nn > 0 else np.zeros(M.shape[0], dtype=np.float32)
X = np.stack([x_of(s) for s in msgs]).astype(np.float32)
print('n=%d zero-rows=%d' % (n, int((np.abs(X).sum(1) < 1e-9).sum())))
MIN_POS, ERA_GAP, MIN_COS, MIN_SEP, N_TARGET = 20, 2000, 0.3, 25, 8
targets = []; last = -10**9
for i in range(MIN_POS, min(MIN_POS + 200, n)):
    sims = X @ X[i]
    far = np.abs(np.arange(n) - i) > ERA_GAP
    js = np.where(far)[0]
    jj = js[int(np.argmax(sims[js]))]
    if i < 26 or (targets and i - targets[-1] < 3):
        print('i=%d best_far_cos=%.3f (j=%d) pass=%s' % (i, sims[jj], jj, sims[jj] >= MIN_COS))
    if sims[jj] < MIN_COS: continue
    if i - last < MIN_SEP: continue
    targets.append(i); last = i
    if len(targets) >= N_TARGET: break
print('targets found:', len(targets), targets[:8])
