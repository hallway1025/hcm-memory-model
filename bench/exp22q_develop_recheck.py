# -*- coding: utf-8 -*-
'''exp22q_develop_recheck.py -- 正确判据(obs_of)重验 exp22 develop 臂（L-035 伪影排查）。
exp22 用 idx==i 旧判据（伪影 #1），develop 臂 0/200 可能是评测伪影；
但 synthetic exact 0/40 若写满(E.n==2000)则是真坏。双判据 + E.n 记录定案。
臂：A0 theta0 / A1 develop eta.05 e6 / A2 +rel.2 / A3 +k328。
语料 peer-01 train 前 3000 条；探针 120（iso<0.5 且已写入）；deg 0.3/0.5。'''
import csv, json, os, re, sys, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))
from hcm import HCM, Config
from hcm.ops import unit

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = 512; SESS = 'peer-01'; N = 3000; NPROBE = 120
NANO = re.compile(r'\[[^\]]{1,8}\]')
def _clean(t): return NANO.sub('', re.sub(r'\s+', '', t))

with open(os.path.join(ROOT,'bench','corpus','train__%s.csv' % SESS), encoding='utf-8-sig', newline='') as f:
    rd = csv.reader(f); next(rd)
    texts = [r[2] for r in rd][:N]
V = sorted({ch for t in texts for ch in _clean(t)} | {'\ufffd'})
VOC = {ch: j for j, ch in enumerate(V)}
rng = np.random.default_rng(7)
W = rng.standard_normal((D, len(V)), dtype=np.float32)
W /= np.linalg.norm(W, axis=0, keepdims=True) + 1e-9
def enc_chars(chars):
    idx = [VOC[c] for c in chars] if chars else [VOC['\ufffd']]
    return unit(W[:, idx].sum(1))
def encode(t): return enc_chars(list(_clean(t)))
def encode_degrade(t, frac, seed):
    r = np.random.default_rng(seed)
    chs = list(_clean(t))
    if len(chs) > 1:
        for j in r.choice(len(chs), size=max(1, int(len(chs)*frac)), replace=False):
            chs[j] = ''
    return enc_chars([c for c in chs if c])

X = np.asarray([encode(t) for t in texts], dtype=np.float32)
def char_ov(a, b):
    sa, sb = set(_clean(a)), set(_clean(b))
    return len(sa & sb) / max(1, len(sa)) if sa else 0.0
t0 = time.time(); iso = []
for i in range(N):
    best = 0.0
    for j in range(N):
        if j != i:
            v = char_ov(texts[i], texts[j])
            if v > best: best = v
    iso.append(best)
print('iso done %.1fs' % (time.time()-t0), flush=True)

def homeo_develop(m, Xs, eta=0.05, epochs=6, n=2000):
    U = [unit(Xs[i]) for i in range(min(n, len(Xs)))]
    for _ in range(epochs):
        for u in U:
            h = m.enc.encode(u)
            m.enc.theta += (eta * (h - m.enc.target)).astype(np.float32)
    m.enc.homeo = True

ARMS = [('A0_theta0', False, None, 82), ('A1_develop', True, None, 82),
        ('A2_develop_rel.2', True, 0.2, 82), ('A3_develop_k328', True, None, 328)]
out = []
for tag, homeo, rel, k in ARMS:
    ta = time.time()
    m = HCM(Config(d_x=D, d_c=D, a=k/4096.0, capacity=N+10, beta=100.0, L=3,
                   use_resid=False, homeo=homeo, rel=rel, seed=0))
    if homeo: homeo_develop(m, X)
    for i in range(N): m.observe(X[i])
    n_written = int(m.E.n)
    obs_of = {s: int(m.E.t[s]) for s in range(n_written)}
    wset = set(obs_of.values())
    cand = [i for i in range(N) if iso[i] < 0.5 and i in wset][:NPROBE]
    rows = []
    for lv in (0.3, 0.5):
        ok_new = ok_old = abst = 0
        for j, i in enumerate(cand):
            q = encode_degrade(texts[i], lv, 7000+j)
            o = m.query(q, do_rif=False)
            ok_new += int(obs_of.get(int(o['idx']), -1) == i)
            ok_old += int(int(o['idx']) == i)
            abst += int(o['action'] == 'abstain')
        rows.append(dict(level=lv, n=len(cand), hit_new=ok_new, hit_old=ok_old, abstain=abst))
        print('%s deg%.1f new %d/%d (old %d) abst=%d n_written=%d' % (tag, lv, ok_new, len(cand), ok_old, abst, n_written), flush=True)
    m2 = HCM(Config(d_x=128, d_c=128, a=k/4096.0, capacity=2010, beta=100.0, L=3,
                    use_resid=False, homeo=homeo, rel=rel, seed=0))
    r = np.random.default_rng(9)
    Y = np.asarray([unit(v) for v in r.standard_normal((2000,128))], dtype=np.float32)
    if homeo: homeo_develop(m2, Y, n=500)
    for y in Y: m2.observe(y)
    nw = int(m2.E.n)
    obs2 = {s: int(m2.E.t[s]) for s in range(nw)}
    so = sn = 0
    for i in range(0, 2000, 50):
        o = m2.query(Y[i], do_rif=False)
        so += int(int(o['idx']) == i)
        sn += int(obs2.get(int(o['idx']), -1) == i)
    print('%s synthetic written=%d exact_old=%d/40 exact_new=%d/40' % (tag, nw, so, sn), flush=True)
    out.append(dict(arm=tag, n_written=n_written, rows=rows, synth_written=nw,
                    synth_old=so, synth_new=sn, write_s=round(time.time()-ta, 1)))
    del m, m2
json.dump(dict(meta=dict(sess=SESS, N=N, judge='obs_of (L-035 fixed)'), arms=out),
          open(os.path.join(ROOT, 'results', 'exp22q_develop_recheck.json'), 'w', encoding='utf-8'),
          ensure_ascii=False, indent=1)
print('saved results/exp22q_develop_recheck.json', flush=True)