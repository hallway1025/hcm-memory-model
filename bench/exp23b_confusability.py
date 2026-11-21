# -*- coding: utf-8 -*-
# exp23b v2: 严格 exp23 同口径的 S256 检索失败取证（NANO clean / V1032 / iso 探针 / deg seed 41000+j）
import csv, json, os, re, sys, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))
from hcm import HCM, Config
from hcm.ops import unit
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
N = 2000; SESS = 'peer-01'
NANO = re.compile(r'\[[^\]]{1,8}\]')
def _clean(t): return NANO.sub('', re.sub(r'\s+', '', t))
with open(os.path.join(ROOT,'bench','corpus','train__%s.csv' % SESS), encoding='utf-8-sig', newline='') as f:
    rd = csv.reader(f); next(rd)
    texts = [r[2] for r in rd][:N]
msgs = [_clean(t) for t in texts]
V = sorted({ch for t in msgs for ch in t} | {'\ufffd'})
Vn = len(V); VOC = {ch: j for j, ch in enumerate(V)}
print('V =', Vn, flush=True)
def learn_embedding_ppmi_svd(msgs_in, r=256, win=3):
    C = np.zeros((Vn, Vn), dtype=np.float64)
    for t in msgs_in:
        chs = [VOC[c] for c in t]
        if not chs: continue
        for i, a in enumerate(chs):
            for b in chs[max(0, i-win):i+win+1]:
                C[a, b] += 1.0
    tot = C.sum()
    pw = C.sum(1) / tot
    P = C / tot
    with np.errstate(divide='ignore'):
        pmi = np.log2(P / (pw[:, None] * pw[None, :] + 1e-12) + 1e-12)
    ppmi = np.maximum(0.0, pmi)
    U, S, Vt = np.linalg.svd(ppmi)
    r = min(r, Vn - 1)
    E = (U[:, :r] * np.sqrt(S[:r])).astype(np.float32)
    print('PPMI nnz=%.3f, r=%d' % (float((ppmi > 0).mean()), r), flush=True)
    return E
def random_proj(dr, seed=7):
    W = np.random.default_rng(seed).standard_normal((dr, Vn), dtype=np.float32)
    W /= np.linalg.norm(W, axis=0, keepdims=True) + 1e-9
    return W
def x_of(chars, E_or_W):
    idx = [VOC[c] for c in chars] if chars else [VOC['\ufffd']]
    if E_or_W.shape[1] == Vn: return unit(E_or_W[:, idx].sum(1))
    return unit(E_or_W[idx].sum(0).astype(np.float32))
def encode_with(M, t): return x_of(list(_clean(t)), M)
def encode_deg_with(M, t, frac, seed):
    r = np.random.default_rng(seed)
    chs = list(_clean(t))
    if len(chs) > 1:
        for j in r.choice(len(chs), size=max(1, int(len(chs)*frac)), replace=False):
            chs[j] = ''
    return x_of([c for c in chs if c], M)
t0 = time.time()
iso = []
for i in range(N):
    best = 0.0
    sa = set(msgs[i])
    for j in range(N):
        if j != i:
            v = len(sa & set(msgs[j])) / max(1, len(sa)) if sa else 0.0
            if v > best: best = v
    iso.append(best)
print('iso done %.1fs' % (time.time()-t0), flush=True)
E_sem = learn_embedding_ppmi_svd(msgs)
W_r = random_proj(256)
out = {}
for tag, M in [('R256', W_r), ('S256', E_sem)]:
    dr = M.shape[1] if M.shape[0] == Vn else M.shape[0]
    X = np.asarray([encode_with(M, t) for t in texts], dtype=np.float32)
    m = HCM(Config(d_x=dr, d_c=dr, a=0.02, capacity=N+10, beta=100.0, L=3,
                   use_resid=False, lam=1.0, seed=0))
    for i in range(N): m.observe(X[i])
    nw = int(m.E.n)
    obs_of = {s: int(m.E.t[s]) for s in range(nw)}
    wset = set(obs_of.values())
    slot_of = {o: s for s, o in obs_of.items()}
    cand = [i for i in range(N) if iso[i] < 0.5 and i in wset][:200]
    fails = []
    for lv in (0.3, 0.5):
        ok = 0; fails_lv = []
        for j, i in enumerate(cand):
            o = m.query(encode_deg_with(M, texts[i], lv, 41000+j), do_rif=False)
            idx = int(o['idx']) if o.get('idx') is not None else -1
            hit = obs_of.get(idx, -1) == i
            ok += int(hit)
            if not hit and lv == 0.3:
                ti = slot_of.get(i, None)
                hc = float(m.enc.encode(x_of(list(_clean(texts[i])), M)) @ m.E.M[:, ti]) if ti is not None else float('nan')
                fails_lv.append({'probe': i, 'top1': idx,
                    'incos_top1': round(float(X[idx] @ X[i]), 4) if idx >= 0 else None,
                    'hcos_true': round(hc, 4) if hc == hc else None,
                    'm1': round(float(o.get('m1', float('nan'))), 4),
                    'abst': int(o.get('abstain', o.get('abst', 0)))})
        out.setdefault(tag, {})[('hit@%.1f' % lv)] = '%d/%d' % (ok, len(cand))
        if lv == 0.3: out[tag]['fails_deg0.3'] = fails_lv[:15]
    sims = []
    for i in range(0, N, 5):
        cs = X[i] @ X.T; cs[i] = -1.0
        sims.append(float(cs.max()))
    sims = np.asarray(sims)
    out[tag]['top2sim_mean'] = round(float(sims.mean()), 4)
    out[tag]['top2sim_p90'] = round(float(np.percentile(sims, 90)), 4)
    out[tag]['top2sim_frac_gt_0.2'] = round(float((sims > 0.2).mean()), 4)
    print(tag, json.dumps({k: v for k, v in out[tag].items() if k != 'fails_deg0.3'}, ensure_ascii=False), flush=True)
json.dump(out, open(os.path.join(ROOT, 'results', 'exp23b_confusability.json'), 'w'), ensure_ascii=False, indent=1)
print('saved results/exp23b_confusability.json', flush=True)