# -*- coding: utf-8 -*-
# exp23c v2 rerun (2026-11-09): B1 数据规模判决（严格 exp23 口径）：嵌入语料 2k(peer-02) vs 全部 6 会话 v2；协议与 v1 逐项相同仅换语料；v1 工件备份 exp23c_data_scale_oldcorpus.json
import csv, glob
import csv, json, os, re, sys, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))
from hcm import HCM, Config
from hcm.ops import unit
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
N = 2000; SESS = 'peer-02'  # v2（D-053）：v1 旧名 peer-01 语料为非现役残留，主会话改 v2
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
        chs = [VOC[c] for c in t if c in VOC]
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
TRAIN_FILES = ['peer-02', 'group-02', 'group-01', 'group-04', 'group-03', 'group-05']  # v2 显式列表：禁 glob——旧 v1 残留（train__peer-01/train__ _）会被 glob 双吃（审计 2026-11-09）
co_texts = []
for p in TRAIN_FILES:
    with open(os.path.join(ROOT, 'bench', 'corpus', 'train__%s.csv' % p), encoding='utf-8-sig', newline='') as f:
        rd = csv.reader(f); next(rd)
        co_texts.extend(r[2] for r in rd)
co_msgs = [_clean(t) for t in co_texts]
print('co texts: %d' % len(co_msgs), flush=True)
E_2k = learn_embedding_ppmi_svd(msgs)
E_full = learn_embedding_ppmi_svd(co_msgs)
ARMS = [('S256_2k', E_2k), ('S256_full', E_full)]
res = {}
for tag, M in ARMS:
    dr = M.shape[1]
    X = np.asarray([encode_with(M, t) for t in texts], dtype=np.float32)
    m = HCM(Config(d_x=dr, d_c=dr, a=0.02, capacity=N+10, beta=100.0, L=3,
                   use_resid=False, lam=1.0, seed=0))
    for i in range(N): m.observe(X[i])
    nw = int(m.E.n)
    obs_of = {s: int(m.E.t[s]) for s in range(nw)}
    wset = set(obs_of.values())
    cand = [i for i in range(N) if iso[i] < 0.5 and i in wset][:200]
    ret = []
    for lv in (0.3, 0.5):
        ok = 0
        for j, i in enumerate(cand):
            o = m.query(encode_deg_with(M, texts[i], lv, 41000+j), do_rif=False)
            ok += int(obs_of.get(int(o['idx']), -1) == i)
        ret.append({'level': lv, 'n': len(cand), 'hit': ok})
        print('%s deg%.1f hit %d/%d' % (tag, lv, ok, len(cand)), flush=True)
    from exp23_pairs import SYN_NEAR, SYN_DISJOINT
    def measure(pairs):
        cs = []; per = []; ins = []
        for a, b in pairs:
            if not (set(_clean(a)) <= set(V) and set(_clean(b)) <= set(V)): continue
            ha = m.enc.encode(x_of(list(_clean(a)), M))
            hb = m.enc.encode(x_of(list(_clean(b)), M))
            c = float(ha @ hb); cs.append(c); per.append(round(c, 4))
            ins.append(round(float(x_of(list(_clean(a)), M) @ x_of(list(_clean(b)), M)), 4))
        return float(np.mean(cs)), per, ins
    syn_m, syn_per, syn_in = measure(SYN_NEAR)
    dis_m, dis_per, dis_in = measure(SYN_DISJOINT)
    res[tag] = {'n_written': nw, 'syn_mean_cos': round(syn_m, 4), 'dis_mean_cos': round(dis_m, 4),
                'dis_input_cos': round(float(np.mean(dis_in)), 4), 'retrieval': ret,
                'dis_per_pair': dis_per, 'dis_input_per_pair': dis_in}
    print(tag, json.dumps({k: v for k, v in res[tag].items() if 'per_pair' not in k}, ensure_ascii=False), flush=True)
d2 = np.asarray(res['S256_2k']['dis_per_pair']); dfull = np.asarray(res['S256_full']['dis_per_pair'])
diff = dfull - d2
tt = float(diff.mean()/(diff.std(ddof=1)/np.sqrt(len(diff)))) if diff.std() > 1e-9 else float('nan')
res['meta'] = {'corpus_version': 'v2', 'sess_2k': SESS, 'train_files': TRAIN_FILES, 'n_full': len(co_msgs), 'note': 'v1 工件=exp23c_data_scale_oldcorpus.json；verdict 沿用 v1 disjoint 判据（docstring 预登记为 near 对，判据不一致已登记 2026-11-09）'}
res['VERDICT'] = {'dis_mean_2k': res['S256_2k']['dis_mean_cos'], 'dis_mean_full': res['S256_full']['dis_mean_cos'],
    'paired_mean': round(float(diff.mean()), 4), 'paired_t': round(tt, 3), 'n': int(len(diff)),
    'gate_2k': res['S256_2k']['retrieval'], 'gate_full': res['S256_full']['retrieval']}
json.dump(res, open(os.path.join(ROOT, 'results', 'exp23c_data_scale.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)  # L-059: Windows 默认 GBK，会话名含 emoji 会炸
print('VERDICT', json.dumps(res['VERDICT'], ensure_ascii=False), flush=True)
print('saved results/exp23c_data_scale.json', flush=True)