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
def learn_embedding_ppmi_svd(msgs_in, r=256, win=3, guard=False):
    C = np.zeros((Vn, Vn), dtype=np.float64)
    for t in msgs_in:
        chs = [VOC[c] for c in t if (c in VOC or not guard)]
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
# 随机回退：语料零支撑字符（含 \ufffd）用确定性随机单位向量（exp23e）
zero_rows = [c for c in V if not any(c in t for t in msgs)]
rng_fb = np.random.default_rng(99)
FB = {c: unit(rng_fb.standard_normal(256).astype(np.float32)) for c in zero_rows}  # 维度=嵌入 r
print('zero-corpus-support chars: %d (ufffd in it: %s)' % (len(zero_rows), '\ufffd' in zero_rows), flush=True)
emoji_n = sum(1 for t in msgs if t == '')
print('emoji-only (cleaned empty) in N=2000: %d' % emoji_n, flush=True)

_xof_default = x_of

def gate(M, fallback=False):
    dr = M.shape[1]
    if fallback:
        def x_of_fb(chars, E_or_W):
            chs = chars if chars else ['\ufffd']
            acc = None
            for c in chs:
                if c in FB:
                    v = FB[c]
                else:
                    j = VOC[c]
                    v = E_or_W[:, j] if E_or_W.shape[1] == Vn else E_or_W[j]
                acc = v.copy() if acc is None else acc + v
            return unit(acc)
        x_of = x_of_fb
        encode_with_f = lambda M2, t2: x_of(list(_clean(t2)), M2)
        encode_deg_f = lambda M2, t2, frac, sd: (lambda r2, chs: ([(chs.__setitem__(j, '') for j in r2.choice(len(chs), size=max(1, int(len(chs)*frac)), replace=False))] and [x_of([c for c in chs if c], M2)])[0] if len(chs) > 1 else x_of(chs, M2))(np.random.default_rng(sd), list(_clean(t2)))
    else:
        x_of = _xof_default
        encode_with_f = lambda M2, t2: x_of(list(_clean(t2)), M2)
        encode_deg_f = encode_deg_with
    X = np.asarray([encode_with_f(M, t) for t in texts], dtype=np.float32)
    m = HCM(Config(d_x=dr, d_c=dr, a=0.02, capacity=N+10, beta=100.0, L=3, use_resid=False, lam=1.0, seed=0))
    for i in range(N): m.observe(X[i])
    nw = int(m.E.n)
    obs_of = {s: int(m.E.t[s]) for s in range(nw)}
    wset = set(obs_of.values())
    pool = [i for i in range(N) if iso[i] < 0.5 and i in wset]
    cand = pool[:200]
    ce = [i for i in pool if msgs[i] != ''][:200]
    ret = {'n_written': nw, 'n_cand': len(cand), 'n_cand_content': len(ce)}
    for tag2, lst in (('hit', cand), ('content_hit', ce)):
        for lv in (0.3, 0.5):
            ok = 0
            for j, i in enumerate(lst):
                o = m.query(encode_deg_f(M, texts[i], lv, 41000+j), do_rif=False)
                ok += int(obs_of.get(int(o['idx']), -1) == i)
            ret['%s@%.1f' % (tag2, lv)] = '%d/%d' % (ok, len(lst))
    return ret

out = {}
r1 = gate(E_sem, fallback=False)
out['S256_2k_asis'] = r1; print('S256_2k asis', json.dumps(r1), flush=True)
r2 = gate(E_sem, fallback=True)
out['S256_2k_randfallback'] = r2; print('S256_2k randfallback', json.dumps(r2), flush=True)
json.dump(out, open(os.path.join(ROOT, 'results', 'exp23e_emoji_verdict.json'), 'w'), ensure_ascii=False, indent=1)
print('saved results/exp23e_emoji_verdict.json', flush=True)