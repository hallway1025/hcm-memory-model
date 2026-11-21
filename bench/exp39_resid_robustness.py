# -*- coding: utf-8 -*-
'''exp39_resid_robustness.py —— 残差增益与 β 坍缩律的跨架构稳健性（exp39，第六轮；预登记）。

目的（外审 #2 问题）：残差纠正与 β 坍缩律是"自家模型内的发现"还是跨架构稳健的性质？
跨三个架构旋钮扫描：(1) 编码器（R512 随机投影 vs S256 语义嵌入）(2) 稀疏度 a（0.01/0.02/0.04，
对应 k=d_h*a）(3) 库规模 N（4000 vs 9033）。

Part A 残差优势网格：24 runs = {R512,S256}×{0.01,0.02,0.04}×{4000,9033}×{resid on,off}
  每run：Config(d_x,d_c=d_x,a=a,capacity=N+10,beta=100,L=3,use_resid=resid,seed=0)，
  m.observe 全存 N 条，100 个干净探针（±200 窗口字符重叠<0.5）deg=0.3，hit@1（obs_of 判据）。
Part B β 坍缩：每编码器一次写入（a=0.02,N=9033,resid=False），β∈{5,10,20,50,100,200} 探针
  （检索期直接传 beta，无需重写），β* = hit>=0.90 的最大 β；跨编码器比较 β*。

预登记门：
  G-R1 N=9033 残差优势 >=0 于 >=5/6 个 (enc,a) 格 且均值 >0
  G-R2 N=4000 平均 |优势| <= 0.10（小库中性，与 exp14 一致）
  G-R3 β* 跨编码器比值 <= 2
诚实条款：残差优势在小库可能为 0（exp14 已证 N=4000 CI 重叠）——如实记录。
确定性：PYTHONHASHSEED=0；OMP=2。输出 results/exp39_resid_robustness.json（smoke 后缀 _smoke）。
'''
import csv, json, os, re, sys, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))
from hcm import HCM, Config
from hcm.ops import unit

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, 'results')
SESS = "peer-02"
SMOKE = os.environ.get('EXP39_SMOKE') == '1'

def load(split):
    p = os.path.join(ROOT, 'bench', 'corpus', '%s__%s.csv' % (split, SESS))
    with open(p, encoding='utf-8-sig', newline='') as f:
        rd = csv.reader(f); next(rd)
        return [r[2] for r in rd]

_npz = np.load(os.path.join(RES, 'exp23_semantic_embed.npz'), allow_pickle=True)
_E = _npz['E']; _chars = [str(c) for c in _npz['chars']]
if '\ufffd' not in _chars:
    _MS = np.hstack([_E.T.copy(), np.random.default_rng(99).standard_normal((_E.shape[1], 1)).astype(np.float32)])
    _MS[:, -1] /= np.linalg.norm(_MS[:, -1])
    _chars = _chars + ['\ufffd']
else:
    _MS = _E.T.copy()
VS = {c: i for i, c in enumerate(_chars)}
NANO = re.compile(r'\[[^\]]{1,8}\]')
def _clean(s): return NANO.sub('', re.sub(r'\s+', '', s))

def enc_S(text):
    idx = [VS.get(c, VS['\ufffd']) for c in _clean(text)] or [VS['\ufffd']]
    v = _MS[:, idx].sum(1); n = np.linalg.norm(v)
    return (v / n).astype(np.float32) if n > 0 else np.zeros(_MS.shape[0], dtype=np.float32)

# R512 encoder (exp21 exact: W columns normalized, seeded 7)
Vr = sorted({ch for t in texts_all for ch in _clean(t)} | {'\ufffd'}) if False else None  # set in main
WR = None
def make_R512(all_texts):
    global Vr, WR
    Vr = sorted({ch for t in all_texts for ch in _clean(t)} | {'\ufffd'})
    Vi_r = {ch: j for j, ch in enumerate(Vr)}
    rng = np.random.default_rng(7)
    WR = rng.standard_normal((512, len(Vr)), dtype=np.float32)
    WR /= np.linalg.norm(WR, axis=0, keepdims=True) + 1e-9
    return Vi_r
def enc_R(text, Vi_r):
    idx = [Vi_r.get(c, Vi_r['\ufffd']) for c in _clean(text)] or [Vi_r['\ufffd']]
    v = WR[:, idx].sum(1); n = np.linalg.norm(v)
    return (v / n).astype(np.float32) if n > 0 else np.zeros(512, dtype=np.float32)

def degrade_chars(text, frac, seed, encf):
    r = np.random.default_rng(seed)
    chs = list(_clean(text))
    if len(chs) > 1:
        k = max(1, int(len(chs) * frac))
        for j in r.choice(len(chs), size=k, replace=False):
            chs[j] = ''
    return encf(''.join(c for c in chs if c))

def run_one(texts, encf, X, probes, N, a_sp, resid, n_probe):
    cfg = Config(d_x=X.shape[1], d_c=X.shape[1], a=a_sp, capacity=N + 10, beta=100.0, L=3,
                 use_resid=resid, seed=0)
    m = HCM(cfg)
    for i in range(N):
        m.observe(X[i])
    obs_of = {int(m.E.t[j]): j for j in range(int(m.E.n))}
    hit = 0
    for i in probes:
        q = degrade_chars(texts[i], 0.3, 101, encf)
        pre = m.enc.raw(q)
        _, a = m.E.retrieve(m.enc.encode_from_pre(pre), L=3, beta=100.0, dense=pre)
        hit += int(int(np.argmax(a)) == obs_of.get(i, -1))
    return hit, int(m.E.n)

def main():
    t0 = time.time()
    global texts_all
    texts_all = load('train')
    Vi_r = make_R512(texts_all)
    def encf_R(t): return enc_R(t, Vi_r)
    encs = {'R512': encf_R, 'S256': enc_S}
    XS = {name: np.asarray([f(t) for t in texts_all], dtype=np.float32) for name, f in encs.items()}

    # clean probes (shared across encoders: criterion is text-level, cap 100 / smoke 30)
    cap = 30 if SMOKE else 100
    csets = [set(_clean(t)) for t in texts_all]
    probes = []
    for i in range(len(texts_all)):
        lo, hi = max(0, i - 200), min(len(texts_all), i + 200)
        ov = max((len(csets[i] & csets[j]) / max(1, min(len(csets[i]), len(csets[j]))))
                 for j in range(lo, hi) if j != i)
        if ov < 0.5: probes.append(i)
        if len(probes) >= cap: break

    rows_A = []
    if SMOKE:
        combos = [('R512', 0.02, 1000), ('S256', 0.02, 1000)]
    else:
        combos = [(e, a, N) for e in ('R512', 'S256') for a in (0.01, 0.02, 0.04) for N in (4000, 9033)]
    for enc_name, a_sp, N in combos:
        h_on, w_on = run_one(texts_all, encs[enc_name], XS[enc_name], probes, N, a_sp, True, cap)
        h_off, w_off = run_one(texts_all, encs[enc_name], XS[enc_name], probes, N, a_sp, False, cap)
        rows_A.append(dict(enc=enc_name, a=a_sp, N=N, hit_on=h_on, hit_off=h_off,
                           written_on=w_on, written_off=w_off,
                           n_probes=len(probes), advantage=h_on - h_off))
        print('A', enc_name, a_sp, N, 'on', h_on, 'off', h_off, 'adv', h_on - h_off,
              '%.0fs' % (time.time() - t0), flush=True)

    rows_B = []
    if SMOKE:
        combos_B = [('R512', 1000)]; betas = (20.0, 100.0)
    else:
        combos_B = [('R512', 9033), ('S256', 9033)]; betas = (5.0, 10.0, 20.0, 50.0, 100.0, 200.0)
    for enc_name, N in combos_B:
        cfg = Config(d_x=XS[enc_name].shape[1], d_c=XS[enc_name].shape[1], a=0.02, capacity=N + 10,
                     beta=100.0, L=3, use_resid=False, seed=0)
        m = HCM(cfg)
        for i in range(N):
            m.observe(XS[enc_name][i])
        obs_of = {int(m.E.t[j]): j for j in range(int(m.E.n))}
        row = dict(enc=enc_name, N=N, written=int(m.E.n), curve={})
        for beta in betas:
            hit = 0
            for i in probes:
                q = degrade_chars(texts_all[i], 0.3, 101, encs[enc_name])
                pre = m.enc.raw(q)
                _, a = m.E.retrieve(m.enc.encode_from_pre(pre), L=3, beta=beta, dense=pre)
                hit += int(int(np.argmax(a)) == obs_of.get(i, -1))
            row['curve']['b%g' % beta] = hit
            print('B', enc_name, 'beta', beta, hit, '%.0fs' % (time.time() - t0), flush=True)
        good = [b for b in betas if row['curve']['b%g' % b] >= 0.9 * len(probes)]
        row['beta_star'] = max(good) if good else 0.0
        rows_B.append(row)

    # gates
    gates = {}
    if not SMOKE:
        big = [r for r in rows_A if r['N'] == 9033]
        small = [r for r in rows_A if r['N'] == 4000]
        adv_big = [r['advantage'] for r in big]
        gates['G_R1'] = dict(advantages=adv_big, n_nonneg=sum(1 for x in adv_big if x >= 0),
                             mean=float(np.mean(adv_big)),
                             pass_=bool(sum(1 for x in adv_big if x >= 0) >= 5 and np.mean(adv_big) > 0))
        adv_small = [r['advantage'] for r in small]
        gates['G_R2'] = dict(mean_abs=float(np.mean(np.abs(adv_small))),
                             pass_=bool(np.mean(np.abs(adv_small)) <= 0.10 * len(probes)))
        bs = [r['beta_star'] for r in rows_B if r['beta_star'] > 0]
        gates['G_R3'] = dict(beta_stars=bs,
                             ratio=(max(bs) / min(bs)) if len(bs) == 2 and min(bs) > 0 else None,
                             pass_=bool(len(bs) == 2 and min(bs) > 0 and max(bs) / min(bs) <= 2.0))
    out = dict(config=dict(session=SESS, probes=len(probes), deg=0.3, beta_probe=100.0, L=3,
                           judge='obs_of (L-035)', smoke=SMOKE),
               part_a=rows_A, part_b=rows_B, gates=gates, elapsed_s=round(time.time() - t0, 1))
    suffix = '_smoke' if SMOKE else ''
    with open(os.path.join(RES, 'exp39_resid_robustness%s.json' % suffix), 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print('GATES', json.dumps(gates, ensure_ascii=False))
    print('EXP39 DONE %.0fs' % (time.time() - t0), flush=True)

if __name__ == '__main__':
    main()
