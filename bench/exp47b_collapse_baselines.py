# -*- coding: utf-8 -*-
'''exp47b_collapse_baselines.py — 跨架构退化-vs-N 形式对比（exp47b；预注册）。
问题：exp47 在 n=9,033 单会话答了 R2 公平基线（BM25/kNN ≈ HCM 身份检索水平，0.24-0.25）；
未答的是「退化-vs-N 的形式对比」——关联记忆（hopf/sdm）随 N 坍缩（exp38: 9k 已 0.01/0.00），
现代检索（BM25/TF-IDF/S256 kNN）是否近似平坦？本实验在 exp45 char 会话的增长轨迹上
逐层（11 个 N 层）测量六臂 hit@1，HCM 参照曲线从 exp45 JSON 提取（不重跑）。

协议（预注册，先于任何数据提交；prereg commit 见 git log）：
- 语料 = results/exp45_s0.json config.corpus_files（6 文件按序拼接，全量 train 行）。
- 全库写入（无写入门；HCM 侧有 dup 门 → 分布等价非项集等价 = 诚实条款③）。
- N 层 = exp45_s0.json beta_star_curve_char 的 11 个 N（运行时读取；超出语料行数则截断）。
- 单次递增写入；每越层：从已写前缀选探针（索引升序扫，±200 窗字符重叠<0.5，取前
  PROBE_CAP 个；层间嵌套），退化 deg∈{0.0,0.3,0.5}，探针 i 用 rng(101+i) 删字符。
- 臂×6：bm25_char1 / bm25_char2 / knn_tfidf / knn_s256（每层对前缀重建）/
  hopf / sdm（增量写入）。
- sdm r 修正（诚实条款②）：exp38 实现为 r=int(mu+2.326*sd)=146 → Binom(256,0.5) 的
  99 分位 → 约 99% 硬位置激活 → cnt 退化为稠密全局求和，其 0/200 是该机制的冻结结果。
  exp47b 主臂用 Kanerva 1988 语义 r=int(mu-2.326*sd)=109（约 1% 激活）。
- 判据 argmax==i，只对已写前缀码本计分（hopf/sdm scores=Xall[:N]@unit(a)）。
- 确定性：PYTHONHASHSEED=0 + BLAS 四 pin=1（启动环境设置）；numpy>=2 用 bitwise_count，
  否则 uint8 LUT 兜底。
- HCM 参照：beta_star_curve_char rates["100"]（固定 beta=100 未补偿轨迹）；判据语义差异
  （argmax==i vs HCM 内容身份 v2）=诚实条款①，比较限形式层（slope）。
预注册终点：
- 主终点 = 每臂 deg0.3 hit@1 vs log10(N) 的 OLS 斜率（每十倍）。
- G-47b.A（锚）：hopf 在最大层 rate < 0.05（exp38 在 9k=0.01）。
- G-47b.B：四现代臂 |slope| < 0.02/decade（近似平坦；违例=现代检索也随 N 退化，如实报）。
- G-47b.C（参照）：HCM beta=100 参照斜率 < 0（坍缩方向；exp45 三 seed 均值）。
- 三门均为「报告式」：任一 FAIL 落账 informative，不构成方法失败（同 exp47 条款）。
诚实条款：①判据语义差异，比较限形式层；②sdm r 修正（上述）；③写入门差异；④探针层间
  嵌套、退化向量 rng(101+i) 固定；⑤sdm n_loc=10*N_final 固定增量（非每层 10*N_k）。
输出 results/exp47b_collapse_baselines.json（smoke 加 _smoke 后缀）。
'''
import csv, json, os, re, sys, time

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('NUMEXPR_NUM_THREADS', '1')
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, 'results')
SMOKE = os.environ.get('EXP47B_SMOKE') == '1'

_npz = np.load(os.path.join(RES, 'exp23_semantic_embed.npz'), allow_pickle=True)
_E = _npz['E']; _chars = [str(c) for c in _npz['chars']]
if '\ufffd' not in _chars:
    _M = np.hstack([_E.T.copy(), np.random.default_rng(99).standard_normal((_E.shape[1], 1)).astype(np.float32)])
    _M[:, -1] /= np.linalg.norm(_M[:, -1])
    _chars = _chars + ['\ufffd']
else:
    _M = _E.T.copy()
Vi = {c: i for i, c in enumerate(_chars)}
D_x = _M.shape[0]
NANO = re.compile(r'\[[^\]]{1,8}\]')

def _clean(s): return NANO.sub('', re.sub(r'\s+', '', s))

def enc(text):
    idx = [Vi.get(c, Vi['\ufffd']) for c in _clean(text)] or [Vi['\ufffd']]
    v = _M[:, idx].sum(1); n = np.linalg.norm(v)
    return (v / n).astype(np.float32) if n > 0 else np.zeros(D_x, dtype=np.float32)

def _degrade_text(text, frac, seed):
    r = np.random.default_rng(seed)
    chs = list(_clean(text))
    if len(chs) > 1:
        k = max(1, int(len(chs) * frac))
        del_set = set(int(j) for j in r.choice(len(chs), size=k, replace=False))
        return ''.join(c for j, c in enumerate(chs) if j not in del_set)
    return ''.join(chs)

def unit(x):
    n = np.linalg.norm(x)
    return x / n if n > 0 else x

_LUT = np.array([bin(i).count('1') for i in range(256)], dtype=np.uint16)
def _popcount_rows(a):
    if a.dtype == np.bool_: a = a.view(np.uint8)
    if hasattr(np, 'bitwise_count'):
        return np.bitwise_count(a).sum(axis=1).astype(np.int64)
    return _LUT[a.view(np.uint8).reshape(len(a), -1)].sum(axis=1, dtype=np.int64)

class FlatCSR:
    def __init__(self, rows_, cols_, vals_, n_rows):
        self.row = np.asarray(rows_, dtype=np.int64)
        self.col = np.asarray(cols_, dtype=np.int64)
        self.val = np.asarray(vals_, dtype=np.float64)
        self.n_rows = n_rows
    def dot(self, q):
        if len(self.val) == 0: return np.zeros(self.n_rows)
        return np.bincount(self.row, weights=self.val * q[self.col], minlength=self.n_rows)

def grams1(s): return list(s)
def grams2(s): return [s[i:i+2] for i in range(len(s)-1)] or [s]

class BM25Arm:
    def __init__(self, dgs, k1=1.5, b=0.75, gfun=grams1):
        self.k1, self.b = k1, b
        self.dgs = dgs
        self.gfun = gfun
        self._build()
    def _build(self):
        dgs = self.dgs
        vocab = {}
        for g in dgs:
            for t in g:
                if t not in vocab: vocab[t] = len(vocab)
        self.vocab = vocab
        V = len(vocab); N = len(dgs)
        rows_ = []; cols_ = []; vals2 = []
        self.dl = np.zeros(N, dtype=np.float64)
        dfc = np.zeros(V)
        for di, g in enumerate(dgs):
            self.dl[di] = len(g)
            tf = {}
            for t in g: tf[t] = tf.get(t, 0) + 1
            for t in set(g): dfc[vocab[t]] += 1
        self.idf = np.log((N - dfc + 0.5) / (dfc + 0.5) + 1.0)
        avgdl = self.dl.mean() if N else 1.0
        for di, g in enumerate(dgs):
            tf = {}
            for t in g: tf[t] = tf.get(t, 0) + 1
            for t, c in tf.items():
                ti = vocab[t]
                w = c * (self.k1 + 1) / (c + self.k1 * (1 - self.b + self.b * self.dl[di] / max(avgdl, 1e-9)))
                rows_.append(di); cols_.append(ti); vals2.append(w * self.idf[ti])
        self.W = FlatCSR(rows_, cols_, vals2, N)
    def scores(self, qtext):
        g = self.gfun(_clean(qtext))
        qv = np.zeros(len(self.vocab))
        for t in g:
            ti = self.vocab.get(t)
            if ti is not None: qv[ti] += 1
        return self.W.dot(qv)

class TfidfArm:
    def __init__(self, dgs):
        self.dgs = dgs
        self._build()
    def _build(self):
        dgs = self.dgs
        vocab = {}
        for g in dgs:
            for t in g:
                if t not in vocab: vocab[t] = len(vocab)
        self.vocab = vocab
        V = len(vocab); N = len(dgs)
        base = []
        dfc = np.zeros(V)
        for di, g in enumerate(dgs):
            tf = {}
            for t in g: tf[t] = tf.get(t, 0) + 1
            for t in set(g): dfc[vocab[t]] += 1
        idf = np.log((1.0 + N) / (1.0 + dfc)) + 1.0
        sq = np.zeros(N)
        for di, g in enumerate(dgs):
            tf = {}
            for t in g: tf[t] = tf.get(t, 0) + 1
            s = 0.0
            for t, c in tf.items():
                b = c * idf[vocab[t]]
                base.append((di, vocab[t], b)); s += b * b
            sq[di] = s
        rows_ = []; cols_ = []; vals2 = []
        for di, ti, b in base:
            rows_.append(di); cols_.append(ti)
            vals2.append(b / max(np.sqrt(sq[di]), 1e-12))
        self.W = FlatCSR(rows_, cols_, vals2, N)
    def scores(self, qtext):
        g = list(_clean(qtext))
        tf = {}
        for t in g: tf[t] = tf.get(t, 0) + 1
        qv = np.zeros(len(self.vocab))
        for t, c in tf.items():
            ti = self.vocab.get(t)
            if ti is not None: qv[ti] = c
        n = np.linalg.norm(qv)
        if n > 0: qv /= n
        return self.W.dot(qv)

class HopfArm:
    def __init__(self):
        self.Mm = np.zeros((D_x, D_x), dtype=np.float32)
    def write(self, x): self.Mm += np.outer(x, x)
    def scores(self, q, X): return X @ unit(self.Mm @ q)

class SDMArm:
    def __init__(self, n_loc, seed=11):
        rng = np.random.default_rng(seed)
        self.n_loc = n_loc
        HL = rng.random((n_loc, D_x)) > 0.5
        self.HLp = np.ascontiguousarray(np.packbits(HL, axis=1)).view(np.uint64)
        self.cnt = np.zeros((n_loc, D_x), dtype=np.float32)
        self.r = int(D_x * 0.5 - 2.326 * (D_x * 0.25) ** 0.5)
    def _addr(self, x): return np.ascontiguousarray(np.packbits(x > 0)).view(np.uint64)
    def write(self, x):
        hc = _popcount_rows(self.HLp ^ self._addr(x))
        act = hc <= self.r
        if act.any(): self.cnt[act] += x
    def scores(self, q, X):
        hc = _popcount_rows(self.HLp ^ self._addr(q))
        act = hc <= self.r
        a = self.cnt[act].sum(0) if act.any() else np.zeros(D_x, dtype=np.float32)
        return X @ unit(a.astype(np.float32))

def main():
    t0 = time.time()
    cfg45 = json.load(open(os.path.join(RES, 'exp45_s0.json'), encoding='utf-8'))
    files = cfg45['config']['corpus_files']
    texts = []
    for fn in files:
        p = os.path.join(ROOT, 'bench', 'corpus', fn)
        with open(p, encoding='utf-8-sig', newline='') as f:
            rd = csv.reader(f); next(rd)
            texts.extend([r[2] for r in rd])
    if SMOKE:
        texts = texts[:400]
        tiers = [100, 200, 400]
        pcap = 20
    else:
        tiers = [int(n['N']) for n in cfg45['beta_star_curve_char']]
        pcap = 200
    n_all = len(texts)
    tiers = sorted(set(min(t, n_all) for t in tiers))
    print('corpus rows=%d files=%d tiers=%s pcap=%d' % (n_all, len(files), tiers, pcap), flush=True)

    Xall = np.asarray([enc(t) for t in texts], dtype=np.float32)
    csets = [set(_clean(t)) for t in texts]
    g1 = [grams1(_clean(t)) for t in texts]
    g2 = [grams2(_clean(t)) for t in texts]
    print('enc+grams done %.0fs' % (time.time() - t0), flush=True)

    n_final = tiers[-1]
    hopf = HopfArm()
    sdm = SDMArm(10 * n_final)
    arm_names = ['bm25_char1', 'bm25_char2', 'knn_tfidf', 'knn_s256', 'hopf', 'sdm']
    rows = {a: [] for a in arm_names}
    wptr = 0
    for N in tiers:
        wt = time.time()
        for i in range(wptr, N):
            hopf.write(Xall[i]); sdm.write(Xall[i])
        wptr = max(wptr, N)
        probes = []
        for i in range(N):
            lo, hi = max(0, i - 200), min(N, i + 200)
            ok = True
            ci = csets[i]
            for j in range(lo, hi):
                if j == i: continue
                if len(ci & csets[j]) / max(1, min(len(ci), len(csets[j]))) >= 0.5:
                    ok = False; break
            if ok: probes.append(i)
            if len(probes) >= pcap: break
        Xp = Xall[:N]
        b1 = BM25Arm(g1[:N]); b2 = BM25Arm(g2[:N], gfun=grams2); tf = TfidfArm(g1[:N])
        acc = {a: {d: [0, 0] for d in ('0.0', '0.3', '0.5')} for a in arm_names}
        for i in probes:
            for deg in (0.0, 0.3, 0.5):
                qt = _degrade_text(texts[i], deg, 101 + i)
                qv = enc(qt)
                scs = {'bm25_char1': b1.scores(qt), 'bm25_char2': b2.scores(qt),
                       'knn_tfidf': tf.scores(qt), 'knn_s256': Xp @ qv,
                       'hopf': hopf.scores(qv, Xp), 'sdm': sdm.scores(qv, Xp)}
                for a, sc in scs.items():
                    k = int(np.argmax(sc))
                    acc[a]['%.1f' % deg][0] += int(k == i)
                    acc[a]['%.1f' % deg][1] += 1
        for a in arm_names:
            ent = {'n': N}
            for d in ('0.0', '0.3', '0.5'):
                h, u = acc[a][d]
                ent['deg' + d] = dict(hit=h, used=u, rate=h / max(1, u))
            rows[a].append(ent)
        print('tier N=%d probes=%d write+eval %.0fs (total %.0fs)' % (N, len(probes), time.time() - wt, time.time() - t0), flush=True)

    xs = np.log10(np.array(tiers, dtype=np.float64))
    def ols(rate_key):
        out = {}
        for a in arm_names:
            ys = np.array([r[rate_key]['rate'] for r in rows[a]], dtype=np.float64)
            if len(xs) > 1 and xs.std() > 0:
                xm, ym = xs.mean(), ys.mean()
                out[a] = float(((xs - xm) * (ys - ym)).sum() / ((xs - xm) ** 2).sum())
            else:
                out[a] = None
        return out
    slopes = ols('deg0.3')
    slopes_d05 = ols('deg0.5')

    hcm_curves = []
    for s in (0, 1, 2):
        try:
            d = json.load(open(os.path.join(RES, 'exp45_s%d.json' % s), encoding='utf-8'))
            cur = d['beta_star_curve_char']
            xs2 = np.log10(np.array([n['N'] for n in cur], dtype=np.float64))
            ys2 = np.array([n['rates']['100'] for n in cur], dtype=np.float64)
            xm, ym = xs2.mean(), ys2.mean()
            hcm_curves.append(float(((xs2 - xm) * (ys2 - ym)).sum() / ((xs2 - xm) ** 2).sum()))
        except Exception as e:
            print('exp45_s%d ref skipped: %s' % (s, e), flush=True)
    hcm_ref_slope = float(np.mean(hcm_curves)) if hcm_curves else None

    gates = dict(
        g47b_a_hopf_max_tier_lt_005=bool(rows['hopf'][-1]['deg0.3']['rate'] < 0.05),
        g47b_b_modern_flat_abs_slope_lt_002_per_decade=bool(
            all(slopes[a] is not None and abs(slopes[a]) < 0.02 for a in arm_names[:4])),
        g47b_c_hcm_ref_slope_negative=bool(hcm_ref_slope is not None and hcm_ref_slope < 0),
    )
    out = dict(exp='exp47b_collapse_baselines', smoke=SMOKE, pre_registered=True,
               config=dict(n_corpus=n_all, corpus_files=files, tiers=tiers, probe_cap=pcap,
                           judge='argmax==i written-prefix', degs=[0.0, 0.3, 0.5],
                           sdm_r=109, sdm_n_loc=10 * n_final,
                           sdm_r_note='exp38 implemented r=146 (~99% activation, degenerate dense-sum); exp47b primary r=109 (1%, Kanerva 1988)',
                           determinism='PYTHONHASHSEED=0 + BLAS pin=1 (launch env)'),
               rows=rows, slopes_deg03=slopes, slopes_deg05=slopes_d05,
               hcm_ref_slope_per_decade_beta100=hcm_ref_slope,
               hcm_ref_curves=hcm_curves, gates=gates, report_only=True,
               elapsed_s=round(time.time() - t0, 1))
    suffix = '_smoke' if SMOKE else ''
    with open(os.path.join(RES, 'exp47b_collapse_baselines%s.json' % suffix), 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print('SLOPES deg0.3/decade:', json.dumps(slopes), flush=True)
    print('HCM ref slope (beta=100):', hcm_ref_slope, 'curves:', hcm_curves, flush=True)
    print('GATES:', json.dumps(gates), flush=True)
    print('EXP47B DONE %.0fs' % (time.time() - t0), flush=True)

if __name__ == '__main__':
    main()
