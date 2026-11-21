# -*- coding: utf-8 -*-
'''exp38_baselines.py —— 公认基线对比（exp38，第六轮；预登记）。

动机（论文外审 #1 问题）：HCM 与经典联想记忆（Hopfield / Kanerva SDM）的区别是什么？
同一数据、同一编码（S256, exp23 npz）、同一退化查询向量、同一种子，三臂对比：

  hopf 经典 Hopfield：M += x x^T；检索 a = M q；判据 argmax_i cos(x_i, a)
  sdm  Kanerva SDM（1988）：N_loc=10*n_lib 个随机 ±1 硬位置，码 sign 二值化地址，
       写：Hamming<=r 的硬位置累加实值码；读：激活位置计数和；r = 二项(256,0.5) 1% 上分位
  hcm  本模型（exp27 同配置同写路径）：m.observe(x) 全存；检索 pre=m.enc.raw(q),
       a = m.E.retrieve(m.enc.encode_from_pre(pre), L=3, beta=100, dense=pre)[1]
       判据 obs_of（L-035）={int(m.E.t[j]): j}；beta=100（D-035）；theta 钉 0（D-038）

任务：
  T1 补全：干净探针（±200 窗口字符重叠<0.5，<=200 个），deg∈{0.3,0.5}，hit@1
  T2 时序链 null：相邻对（gap<=30s、不同发送者、双方已写入、len>=4；exp27 协议），
     next@5 与 ctrl@5（随机已写槽位排名）
  T3 写入门控：HCM 写入数 vs 基线全写

预登记门：
  G-B1 HCM deg0.3 >= max(hopf,sdm)+0.05
  G-B2 HCM deg0.3 >= hopf+0.15 且 >= sdm+0.10
  G-B3 三臂 next@5 均 <= 0.05（exp27 否定基线的跨架构普遍化）
诚实条款：小库低干扰区基线可能不输；全部数字如实入文。
确定性：PYTHONHASHSEED=0；退化向量每 (probe,deg) 只算一次、三臂共用。
输出：results/exp38_baselines.json
'''
import csv, json, os, re, sys, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))
from hcm import HCM, Config
from hcm.ops import unit

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, 'results')
SESS = "peer-02"
SMOKE = os.environ.get('EXP38_SMOKE') == '1'

def load(split):
    p = os.path.join(ROOT, 'bench', 'corpus', '%s__%s.csv' % (split, SESS))
    with open(p, encoding='utf-8-sig', newline='') as f:
        rd = csv.reader(f); next(rd)
        return [(t, s, c) for t, s, c in rd]

_npz = np.load(os.path.join(RES, 'exp23_semantic_embed.npz'), allow_pickle=True)
_E = _npz['E']; _chars = [str(c) for c in _npz['chars']]
if '\ufffd' not in _chars:
    _M = np.hstack([_E.T.copy(), np.random.default_rng(99).standard_normal((_E.shape[1], 1)).astype(np.float32)])
    _M[:, -1] /= np.linalg.norm(_M[:, -1])
    _chars = _chars + ['\ufffd']
else:
    _M = _E.T.copy()
Vi = {c: i for i, c in enumerate(_chars)}
NANO = re.compile(r'\[[^\]]{1,8}\]')
def _clean(s): return NANO.sub('', re.sub(r'\s+', '', s))
def enc(text):
    idx = [Vi.get(c, Vi['\ufffd']) for c in _clean(text)] or [Vi['\ufffd']]
    v = _M[:, idx].sum(1); n = np.linalg.norm(v)
    return (v / n).astype(np.float32) if n > 0 else np.zeros(_M.shape[0], dtype=np.float32)
D_x = _M.shape[0]

def degrade_chars(text, frac, seed):
    r = np.random.default_rng(seed)
    chs = list(_clean(text))
    if len(chs) > 1:
        k = max(1, int(len(chs) * frac))
        for j in r.choice(len(chs), size=k, replace=False):
            chs[j] = ''
    return enc(''.join(c for c in chs if c))

def binom_ppf_hi(d, p):
    mu, sd = d * 0.5, (d * 0.25) ** 0.5
    return int(mu + 2.326 * sd)

# ---------------- arms ----------------
class HopfArm:
    def __init__(self):
        self.Mm = np.zeros((D_x, D_x), dtype=np.float32)
    def write(self, x): self.Mm += np.outer(x, x)
    def scores(self, q):
        a = self.Mm @ q
        return Xall @ unit(a.astype(np.float32))   # score per stored code

class SDMArm:
    def __init__(self, n_lib, seed=11):
        rng = np.random.default_rng(seed)
        self.n_loc = 10 * n_lib
        self.HL = rng.random((self.n_loc, D_x)) > 0.5
        self.cnt = np.zeros((self.n_loc, D_x), dtype=np.float32)
        self.r = binom_ppf_hi(D_x, 0.01)
    def write(self, x):
        addr = x > 0
        act = (self.HL != addr).sum(1) <= self.r
        if act.any(): self.cnt[act] += x
    def scores(self, q):
        addr = q > 0
        act = (self.HL != addr).sum(1) <= self.r
        a = self.cnt[act].sum(0) if act.any() else np.zeros(D_x, dtype=np.float32)
        return Xall @ unit(a.astype(np.float32))

class HCMArm:
    def __init__(self, n_lib):
        cfg = Config(d_x=D_x, d_c=D_x, a=0.02, capacity=n_lib + 10, beta=100.0, L=3, use_resid=False, seed=0)
        self.m = HCM(cfg)
    def write(self, x): self.m.observe(x)
    def scores(self, q):
        pre = self.m.enc.raw(q)
        _, a = self.m.E.retrieve(self.m.enc.encode_from_pre(pre), L=3, beta=100.0, dense=pre)
        return np.asarray(a, dtype=np.float32)

# ---------------- main ----------------
def main():
    t0 = time.time()
    rows = load('train')
    if SMOKE: rows = rows[:400]
    times = [r[0] for r in rows]; senders = [r[1] for r in rows]
    texts = [r[2] for r in rows]
    n_lib = len(texts)
    global Xall
    Xall = np.asarray([enc(t) for t in texts], dtype=np.float32)

    # T1 probes: clean (char overlap < 0.5 within +-200 window), cap 200 (smoke 30)
    cap = 30 if SMOKE else 200
    csets = [set(_clean(t)) for t in texts]
    probes = []
    for i in range(n_lib):
        lo, hi = max(0, i - 200), min(n_lib, i + 200)
        ov = max((len(csets[i] & csets[j]) / max(1, min(len(csets[i]), len(csets[j]))))
                 for j in range(lo, hi) if j != i)
        if ov < 0.5: probes.append(i)
        if len(probes) >= cap: break

    # T2 pairs: exp27 protocol
    from datetime import datetime
    def tparse(s):
        try: return datetime.fromisoformat(s.replace('/', '-'))
        except Exception: return None
    pairs = []
    for i in range(n_lib - 1):
        if i + 1 >= n_lib: break
        a, b = _clean(texts[i]), _clean(texts[i + 1])
        if len(a) < 4 or len(b) < 4 or senders[i] == senders[i + 1]: continue
        ta, tb = tparse(times[i]), tparse(times[i + 1])
        if ta is None or tb is None or (tb - ta).total_seconds() > 30: continue
        pairs.append(i)
    npair = 30 if SMOKE else 200
    rngp = np.random.default_rng(2700); rngp.shuffle(pairs); pairs = pairs[:npair]

    arms = {}
    arms['hopf'] = HopfArm()
    arms['sdm'] = SDMArm(n_lib)
    arms['hcm'] = HCMArm(n_lib)
    # write phase (same order)
    for name, arm in arms.items():
        wt = time.time()
        for i in range(n_lib): arm.write(Xall[i])
        print('%s written %d/%d %.0fs' % (name, (n_lib if name != 'hcm' else int(arm.m.E.n)), n_lib, time.time() - wt), flush=True)
    obs_of = {int(arms['hcm'].m.E.t[j]): j for j in range(int(arms['hcm'].m.E.n))}

    # scores cache per arm for probes and pairs (recompute per query — states differ)
    res = {}
    for name, arm in arms.items():
        r = dict(written=(n_lib if name != 'hcm' else int(arm.m.E.n)))
        for deg in (0.3, 0.5):
            hit = 0
            for i in probes:
                q = degrade_chars(texts[i], deg, 101)
                sc = arm.scores(q)
                k = int(np.argmax(sc))
                if name == 'hcm':
                    hit += int(k == obs_of.get(i, -1))
                else:
                    hit += int(k == i)
            r['deg%.1f' % deg] = hit
        # chain
        selfh = next5 = ctrl5 = 0; ranks = []; cranks = []
        used = 0
        rngc = np.random.default_rng(2701)
        for i in pairs:
            j = i + 1
            sc = arm.scores(Xall[i])
            if name == 'hcm':
                tgt = obs_of.get(j, -1); src = obs_of.get(i, -1)
                if tgt < 0 or src < 0: continue
            else:
                tgt = j; src = i
            order = np.argsort(-sc)
            k = int(order[0])
            selfh += int(k == src)
            rk = int(np.where(order == tgt)[0][0]); ranks.append(rk)
            if rk < 5: next5 += 1
            # control: random WRITTEN slot
            cslot = tgt
            while True:
                cslot = int(rngc.integers(0, n_lib if name != 'hcm' else int(arm.m.E.n)))
                if name == 'hcm':
                    if cslot not in (src, tgt) and cslot < int(arm.m.E.n): break
                else:
                    if cslot not in (src, tgt): break
            rkc = int(np.where(order == cslot)[0][0]); cranks.append(rkc)
            if rkc < 5: ctrl5 += 1
            used += 1
        r.update(chain_used=used, self_hit1=selfh, next5=next5, ctrl5=ctrl5,
                 next_rank_med=float(np.median(ranks)) if ranks else None,
                 ctrl_rank_med=float(np.median(cranks)) if cranks else None)
        res[name] = r
        print(name, json.dumps(r, ensure_ascii=False), flush=True)

    n_p = max(1, len(probes)); n_ch = max(1, res['hcm']['chain_used'])
    gates = {
        "G_B1": dict(hcm=res['hcm']['deg0.3'] / n_p, best_base=max(res['hopf']['deg0.3'], res['sdm']['deg0.3']) / n_p,
                     delta=res['hcm']['deg0.3'] / n_p - max(res['hopf']['deg0.3'], res['sdm']['deg0.3']) / n_p,
                     pass_=bool(res['hcm']['deg0.3'] / n_p - max(res['hopf']['deg0.3'], res['sdm']['deg0.3']) / n_p >= 0.05)),
        "G_B2": dict(vs_hopf=res['hcm']['deg0.3'] / n_p - res['hopf']['deg0.3'] / n_p,
                     vs_sdm=res['hcm']['deg0.3'] / n_p - res['sdm']['deg0.3'] / n_p,
                     pass_=bool(res['hcm']['deg0.3'] / n_p - res['hopf']['deg0.3'] / n_p >= 0.15 and
                                res['hcm']['deg0.3'] / n_p - res['sdm']['deg0.3'] / n_p >= 0.10)),
        "G_B3": dict(next5_rates={a: round(res[a]['next5'] / max(1, res[a]['chain_used']), 4) for a in res},
                     max_rate=max(res[a]['next5'] / max(1, res[a]['chain_used']) for a in res),
                     pass_=bool(all(res[a]['next5'] / max(1, res[a]['chain_used']) <= 0.05 for a in res))),
    }
    out = dict(config=dict(n_lib=n_lib, probes=len(probes), pairs=len(pairs), D_x=int(D_x),
                           embed='S256', session=SESS, smoke=SMOKE,
                           beta=100.0, L=3, use_resid=False, judge='argmax score; hcm via obs_of (L-035)'),
               rows=res, gates=gates, probes=probes, elapsed_s=round(time.time() - t0, 1))
    suffix = '_smoke' if SMOKE else ''
    with open(os.path.join(RES, 'exp38_baselines%s.json' % suffix), 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print('GATES', json.dumps(gates, ensure_ascii=False))
    print('EXP38 DONE %.0fs' % (time.time() - t0), flush=True)

if __name__ == '__main__':
    main()
