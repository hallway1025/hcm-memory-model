# -*- coding: utf-8 -*-
'''exp47_fair_retrieval.py —— 公平检索基线（exp47；R2 最低清单：BM25 / embedding kNN）。

exp38 证明了经典联想记忆基线（hopf 2/200, sdm 0/200 @deg0.3）；R2 指出还缺现代检索
基线。本实验在 exp38 完全相同的语料/探针/退化/判据上补四臂：

  bm25_char1   BM25（k1=1.5, b=0.75）over 字符 1-gram，查询=退化字符串
  bm25_char2   BM25 over 字符 2-gram（CJK 无分词器标准做法）
  knn_tfidf    字符 1-gram TF-IDF 余弦
  knn_s256     S256 嵌入余弦 kNN（=exp38 hcm 臂的输入表征直接做最近邻，
               回答『同一表征不经过吸引子读出能走多远』）

协议复用（逐项同 exp38）：
  语料 bench/corpus/train__peer-02.csv；探针=±200 窗字符重叠<0.5 cap 200；
  退化=deg {0,0.3,0.5} 每探针 seed 101；判据=argmax(score)==i（全库写入无门控）。
参照系：exp38 同探针 hcm deg0.3=48/200。
输出 results/exp47_fair_retrieval.json（报告式门，无方向风险，两分支均 informative）。
'''
import csv, json, os, re, sys, time
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, 'results')
SESS = "peer-02"
SMOKE = os.environ.get('EXP47_SMOKE') == '1'

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

def _degrade_text(text, frac, seed):
    r = np.random.default_rng(seed)
    chs = list(_clean(text))
    if len(chs) > 1:
        k = max(1, int(len(chs) * frac))
        del_set = set(int(j) for j in r.choice(len(chs), size=k, replace=False))
        return ''.join(c for j, c in enumerate(chs) if j not in del_set)
    return ''.join(chs)

# ---------------- minimal sparse helpers (no scipy dependency) ----------------
class FlatCSR:
    '''COO-backed sparse matrix: W @ q via bincount. Rows=docs, cols=vocab.'''
    def __init__(self, rows_, cols_, vals_, n_rows):
        self.row = np.asarray(rows_, dtype=np.int64)
        self.col = np.asarray(cols_, dtype=np.int64)
        self.val = np.asarray(vals_, dtype=np.float64)
        self.n_rows = n_rows
    def dot(self, q):
        if len(self.val) == 0: return np.zeros(self.n_rows)
        contrib = self.val * q[self.col]
        return np.bincount(self.row, weights=contrib, minlength=self.n_rows)

# ---------------- lexical arms ----------------
class BM25Arm:
    def __init__(self, docs, ngram=1, k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        self.docs = [_clean(d) for d in docs]
        self.ngram = ngram
        self._build()
    def _grams(self, s):
        if self.ngram == 1: return list(s)
        return [s[i:i+2] for i in range(len(s)-1)] or [s]
    def _build(self):
        dgs = [self._grams(d) for d in self.docs]
        vocab = {}
        for g in dgs:
            for t in g:
                if t not in vocab: vocab[t] = len(vocab)
        self.vocab = vocab
        V = len(vocab); N = len(dgs)
        rows_ = []; cols_ = []; vals_ = []
        self.dl = np.zeros(N, dtype=np.float64)
        for di, g in enumerate(dgs):
            self.dl[di] = len(g)
            tf = {}
            for t in g: tf[t] = tf.get(t, 0) + 1
            for t, c in tf.items():
                rows_.append(di); cols_.append(vocab[t]); vals_.append(c)
        # df via unique (doc, term) pairs
        pair = rows_[:0]
        seen = set()
        dfc = np.zeros(V)
        for di, g in enumerate(dgs):
            for t in set(g): dfc[vocab[t]] += 1
        self.idf = np.log((N - dfc + 0.5) / (dfc + 0.5) + 1.0)
        avgdl = self.dl.mean() if N else 1.0
        # BM25 weight per (doc, term): tf*(k1+1)/(tf + k1*(1-b+b*dl/avgdl)) * idf
        vals2 = []
        for di, g in enumerate(dgs):
            tf = {}
            for t in g: tf[t] = tf.get(t, 0) + 1
            for t, c in tf.items():
                ti = vocab[t]
                w = c * (self.k1 + 1) / (c + self.k1 * (1 - self.b + self.b * self.dl[di] / max(avgdl, 1e-9)))
                vals2.append(w * self.idf[ti])
        self.W = FlatCSR(rows_, cols_, vals2, N)
    def scores(self, qtext):
        g = self._grams(_clean(qtext))
        qv = np.zeros(len(self.vocab))
        for t in g:
            ti = self.vocab.get(t)
            if ti is not None: qv[ti] += 1
        return self.W.dot(qv)

class TfidfArm:
    def __init__(self, docs):
        self.docs = [_clean(d) for d in docs]
        self._build()
    def _grams(self, s): return list(s)
    def _build(self):
        dgs = [self._grams(d) for d in self.docs]
        vocab = {}
        for g in dgs:
            for t in g:
                if t not in vocab: vocab[t] = len(vocab)
        self.vocab = vocab
        V = len(vocab); N = len(dgs)
        rows_ = []; cols_ = []; vals_ = []
        for di, g in enumerate(dgs):
            tf = {}
            for t in g: tf[t] = tf.get(t, 0) + 1
            for t, c in tf.items():
                rows_.append(di); cols_.append(vocab[t]); vals_.append(c)
        dfc = np.zeros(V)
        for di, g in enumerate(dgs):
            for t in set(g): dfc[vocab[t]] += 1
        idf = np.log((1.0 + N) / (1.0 + dfc)) + 1.0
        # L2-normalized TF-IDF rows (row-wise norm over nnz entries)
        sq = {}  # (doc) -> sum of squares
        vals2 = []
        for di, g in enumerate(dgs):
            tf = {}
            for t in g: tf[t] = tf.get(t, 0) + 1
            s = 0.0
            for t, c in tf.items():
                s += (c * idf[vocab[t]]) ** 2
            sq[di] = s
        for k_i in range(len(rows_)):
            di = rows_[k_i]; ti = cols_[k_i]
            base = vals_[k_i] * idf[ti]
            vals2.append(base / max(np.sqrt(sq[di]), 1e-12))
        self.W = FlatCSR(rows_, cols_, vals2, N)
    def scores(self, qtext):
        g = self._grams(_clean(qtext))
        tf = {}
        for t in g: tf[t] = tf.get(t, 0) + 1
        qv = np.zeros(len(self.vocab))
        for t, c in tf.items():
            ti = self.vocab.get(t)
            if ti is not None: qv[ti] = c
        n = np.linalg.norm(qv)
        if n > 0: qv /= n
        return self.W.dot(qv)

class KnnS256Arm:
    def __init__(self, X): self.X = X
    def scores(self, q): return self.X @ np.asarray(q, dtype=np.float32)

# ---------------- main ----------------
def main():
    t0 = time.time()
    rows = load('train')
    if SMOKE: rows = rows[:400]
    texts = [r[2] for r in rows]
    n_lib = len(texts)
    Xall = np.asarray([enc(t) for t in texts], dtype=np.float32)

    cap = 30 if SMOKE else 200
    csets = [set(_clean(t)) for t in texts]
    probes = []
    for i in range(n_lib):
        lo, hi = max(0, i - 200), min(n_lib, i + 200)
        ov = max((len(csets[i] & csets[j]) / max(1, min(len(csets[i]), len(csets[j]))))
                 for j in range(lo, hi) if j != i)
        if ov < 0.5: probes.append(i)
        if len(probes) >= cap: break

    arms = {}
    arms['bm25_char1'] = BM25Arm(texts, ngram=1)
    arms['bm25_char2'] = BM25Arm(texts, ngram=2)
    arms['knn_tfidf'] = TfidfArm(texts)
    arms['knn_s256'] = KnnS256Arm(Xall)

    res = {}
    for name, arm in arms.items():
        r = dict(arm=name)
        for deg in (0.0, 0.3, 0.5):
            hit = 0; used = 0
            for i in probes:
                qt = _degrade_text(texts[i], deg, 101)
                if name == 'knn_s256':
                    sc = arm.scores(enc(qt))
                else:
                    sc = arm.scores(qt)
                k = int(np.argmax(sc))
                hit += int(k == i); used += 1
            r['deg%.1f' % deg] = dict(hit=hit, used=used, rate=hit / max(1, used))
        res[name] = r
        print(name, json.dumps(r, ensure_ascii=False), flush=True)

    ref38 = dict(hcm_deg03=48, hopf_deg03=2, sdm_deg03=0, note='exp38 same probes/corpus')
    out = dict(config=dict(n_lib=n_lib, probes=len(probes), embed='S256', session=SESS,
                           smoke=SMOKE, judge='argmax==i full-store',
                           bm25='k1=1.5 b=0.75 char-gram', ref38=ref38),
               rows=res, elapsed_s=round(time.time() - t0, 1))
    suffix = '_smoke' if SMOKE else ''
    with open(os.path.join(RES, 'exp47_fair_retrieval%s.json' % suffix), 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print('EXP47 DONE %.0fs' % (time.time() - t0), flush=True)

if __name__ == '__main__':
    main()