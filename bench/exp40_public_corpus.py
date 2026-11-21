# -*- coding: utf-8 -*-
'''exp40_public_corpus.py —— 公开语料端到端（exp40，第六轮；预登记）。

目的（外审 #3 问题）：HCM 在公开可复现的英文数据上表现如何？数据 = DailyDialog
（Li et al. 2017, 11,118 段双人对话；train 8,894 段/68,994 条，test 2,224 段/18,176 条，
bench/corpus_public/，CC BY 4.0）。全部结果可第三方复现。

编码双臂：
  NEW256  CharacterSemanticEncoder(256,window=3,seed=0)，fit 于 train 全部文本（opt-verify B1 产物）
  R512    exp21 口径随机投影（rng(7) 列归一），词表 = train+test 去重字符

任务：
  T1 补全：N∈{2000,8000,20000} 前缀写入，100 干净探针（±200 窗口字符重叠<0.5），
     deg∈{0.3,0.5}，hit@1（obs_of, L-035）
  T2 时序链 null：相邻话轮对（gap<=30s、不同发送者、双方已写入、len>=4；exp27 协议），
     200 对，next@5 + ctrl@5——exp27 否定基线在英文公开数据上的复现
  T3 弃权分离：100 库内退化探针 vs 100 test（从未写入）消息，raw_evidence AUC
     （exp13 结论：raw_evidence 是可用置信信号）

预登记门：
  G-P1 NEW256 hit@1 >= R512 hit@1（N=8000, deg0.3，同探针配对）
  G-P2 双臂 next@5 <= 0.05（链 null 跨数据集复现）
  G-P3 max 臂 AUC(raw_evidence) >= 0.70
诚实条款：exp21 v2 在中文私聊上 hit@1=0/200（地板）；英文 99 字符词表可能不同——数字如实入文。
确定性：PYTHONHASHSEED=0；OMP=2。输出 results/exp40_public_corpus.json（smoke 后缀 _smoke）。
'''
import csv, json, os, re, sys, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))
from hcm import HCM, Config
from hcm.ops import unit
from hcm.text_encoder import CharacterSemanticEncoder

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, 'results')
SMOKE = os.environ.get('EXP40_SMOKE') == '1'

def load(split):
    p = os.path.join(ROOT, 'bench', 'corpus_public', '%s__dailydialog.csv' % split)
    with open(p, encoding='utf-8-sig', newline='') as f:
        rd = csv.reader(f); next(rd)
        return [(r[0], r[1], r[2]) for r in rd]

NANO = re.compile(r'\[[^\]]{1,8}\]')
def _clean(s): return NANO.sub('', re.sub(r'\s+', ' ', s)).strip()

def degrade_chars(text, frac, seed, encf):
    r = np.random.default_rng(seed)
    chs = list(_clean(text))
    if len(chs) > 1:
        k = max(1, int(len(chs) * frac))
        for j in r.choice(len(chs), size=k, replace=False):
            chs[j] = ''
    return encf(''.join(c for c in chs if c))

def auc_rank(ins, outs):
    '''Mann-Whitney AUC: P(in > out) + 0.5 P(tie)'''
    allv = [(v, 1) for v in ins] + [(v, 0) for v in outs]
    allv.sort(key=lambda x: x[0])
    n_in, n_out = len(ins), len(outs)
    if n_in == 0 or n_out == 0: return None
    rank_sum_in = 0.0; i = 0
    while i < len(allv):
        j = i
        while j + 1 < len(allv) and allv[j + 1][0] == allv[i][0]: j += 1
        avg_rank = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            if allv[k][1] == 1: rank_sum_in += avg_rank
        i = j + 1
    return (rank_sum_in - n_in * (n_in + 1) / 2.0) / (n_in * n_out)

def main():
    t0 = time.time()
    tr = load('train'); te = load('test')
    texts_tr = [r[2] for r in tr]; texts_te = [r[2] for r in te]
    times_tr = [r[0] for r in tr]; senders_tr = [r[1] for r in tr]
    if SMOKE:
        texts_tr = texts_tr[:6000]; times_tr = times_tr[:6000]; senders_tr = senders_tr[:6000]
        fit_texts = texts_tr[:5000]
    else:
        fit_texts = texts_tr
    print('train=%d test=%d' % (len(texts_tr), len(texts_te)), flush=True)

    # encoders
    new = CharacterSemanticEncoder(dim=256, window=3, seed=0)
    ft = time.time(); new.fit(fit_texts)
    print('NEW256 fit %.0fs dim=%d' % (time.time() - ft, new.dim), flush=True)
    # full vocab determinism: use all texts both splits
    Vr = sorted({ch for t in texts_tr for ch in _clean(t)} | {ch for t in texts_te[:4000] for ch in _clean(t)} | {'\ufffd'})
    Vir = {ch: j for j, ch in enumerate(Vr)}
    rng = np.random.default_rng(7)
    WR = rng.standard_normal((512, len(Vr)), dtype=np.float32)
    WR /= np.linalg.norm(WR, axis=0, keepdims=True) + 1e-9
    def enc_new(t): return new.encode(t)
    def enc_r(t):
        idx = [Vir.get(c, Vir['\ufffd']) for c in _clean(t)] or [Vir['\ufffd']]
        v = WR[:, idx].sum(1); n = np.linalg.norm(v)
        return (v / n).astype(np.float32) if n > 0 else np.zeros(512, dtype=np.float32)
    arms = {'NEW256': enc_new, 'R512': enc_r}

    # clean probes (train side; English adaptation of the <0.5 threshold:
    # with a 99-char alphabet nearly all pairs exceed 0.5, so rank by window
    # max-overlap ascending and take the cleanest 100 — same probe set for all arms)
    cap = 30 if SMOKE else 100
    csets = [set(_clean(t)) for t in texts_tr]
    ovs = []
    for i in range(len(texts_tr)):
        lo, hi = max(0, i - 200), min(len(texts_tr), i + 200)
        ov = max((len(csets[i] & csets[j]) / max(1, min(len(csets[i]), len(csets[j]))))
                 for j in range(lo, hi) if j != i)
        ovs.append((ov, i))
    ovs.sort()
    probes = [i for _, i in ovs[:cap]]
    probes.sort()

    Ns = (2000,) if SMOKE else (2000, 8000, 20000)
    results = {'T1': [], 'T2': {}, 'T3': {}}
    for N in Ns:
        Xs = {name: np.asarray([f(t) for t in texts_tr[:N]], dtype=np.float32) for name, f in arms.items()}
        for name in arms:
            cfg = Config(d_x=Xs[name].shape[1], d_c=Xs[name].shape[1], a=0.02, capacity=N + 10,
                         beta=100.0, L=3, use_resid=False, seed=0)
            m = HCM(cfg)
            wt = time.time()
            for i in range(N): m.observe(Xs[name][i])
            obs_of = {int(m.E.t[j]): j for j in range(int(m.E.n))}
            written = int(m.E.n)
            row = dict(arm=name, N=N, written=written, write_s=round(time.time() - wt, 1))
            for deg in (0.3, 0.5):
                hit = 0
                for i in probes:
                    q = degrade_chars(texts_tr[i], deg, 101, arms[name])
                    pre = m.enc.raw(q)
                    _, a = m.E.retrieve(m.enc.encode_from_pre(pre), L=3, beta=100.0, dense=pre)
                    hit += int(int(np.argmax(a)) == obs_of.get(i, -1))
                row['deg%.1f' % deg] = hit
            results['T1'].append(row)
            print('T1', json.dumps(row, ensure_ascii=False), flush=True)

            if N == (2000 if SMOKE else 8000):
                # T2 chain null
                from datetime import datetime
                def tparse(s):
                    try: return datetime.fromisoformat(s.replace('/', '-'))
                    except Exception: return None
                pairs = []
                for i in range(N - 1):
                    a, b = _clean(texts_tr[i]), _clean(texts_tr[i + 1])
                    if len(a) < 4 or len(b) < 4 or senders_tr[i] == senders_tr[i + 1]: continue
                    ta, tb = tparse(times_tr[i]), tparse(times_tr[i + 1])
                    if ta is None or tb is None or (tb - ta).total_seconds() > 30: continue
                    if i not in obs_of or (i + 1) not in obs_of: continue
                    pairs.append(i)
                npair = 50 if SMOKE else 200
                rngp = np.random.default_rng(2700); rngp.shuffle(pairs); pairs = pairs[:npair]
                selfh = next5 = ctrl5 = 0; ranks = []; cranks = []
                rngc = np.random.default_rng(2701)
                for i in pairs:
                    j = i + 1
                    pre = m.enc.raw(Xs[name][i])
                    _, a = m.E.retrieve(m.enc.encode_from_pre(pre), L=3, beta=100.0, dense=pre)
                    order = np.argsort(-np.asarray(a))
                    src, tgt = obs_of[i], obs_of[j]
                    selfh += int(int(order[0]) == src)
                    rk = int(np.where(order == tgt)[0][0]); ranks.append(rk)
                    if rk < 5: next5 += 1
                    cslot = tgt
                    while True:
                        cslot = int(rngc.integers(0, written))
                        if cslot not in (src, tgt): break
                    rkc = int(np.where(order == cslot)[0][0]); cranks.append(rkc)
                    if rkc < 5: ctrl5 += 1
                results['T2'][name] = dict(pairs=len(pairs), self_hit1=selfh,
                                           next5=next5, next_rate=round(next5 / max(1, len(pairs)), 4),
                                           ctrl5=ctrl5, ctrl_rate=round(ctrl5 / max(1, len(pairs)), 4),
                                           next_rank_med=float(np.median(ranks)) if ranks else None,
                                           ctrl_rank_med=float(np.median(cranks)) if cranks else None,
                                           candidate_pairs=len(pairs))
                print('T2', name, json.dumps(results['T2'][name], ensure_ascii=False), flush=True)
                # T3 abstention
                n_ab = 30 if SMOKE else 100
                ins = []
                rnga = np.random.default_rng(3001)
                for i in probes[:n_ab]:
                    q = degrade_chars(texts_tr[i], 0.3, 101, arms[name])
                    qq = m.query(q)
                    ins.append(float(qq['raw_evidence']))
                outs = []
                te_idx = rnga.choice(len(texts_te), size=min(n_ab, len(texts_te)), replace=False)
                for k in te_idx:
                    qq = m.query(arms[name](texts_te[int(k)]))
                    outs.append(float(qq['raw_evidence']))
                results['T3'][name] = dict(auc=auc_rank(ins, outs), in_mean=float(np.mean(ins)),
                                           out_mean=float(np.mean(outs)), n_in=len(ins), n_out=len(outs))
                print('T3', name, json.dumps(results['T3'][name], ensure_ascii=False), flush=True)
            del m

    # gates
    t1 = {(r['arm'], r['N']): r for r in results['T1']}
    Nkey = 2000 if SMOKE else 8000
    gates = {}
    r_new, r_r = t1[('NEW256', Nkey)], t1[('R512', Nkey)]
    gates['G_P1'] = dict(new=r_new['deg0.3'], r512=r_r['deg0.3'],
                         pass_=bool(r_new['deg0.3'] >= r_r['deg0.3']))
    rates = {a: results['T2'][a]['next_rate'] for a in results['T2']}
    gates['G_P2'] = dict(rates=rates, pass_=bool(rates and all(v <= 0.05 for v in rates.values())))
    aucs = {a: results['T3'][a]['auc'] for a in results['T3'] if results['T3'][a]['auc'] is not None}
    gates['G_P3'] = dict(aucs=aucs, pass_=bool(aucs and max(aucs.values()) >= 0.70))
    out = dict(config=dict(data='DailyDialog train (CC BY 4.0)', n_train=len(texts_tr), n_test=len(texts_te),
                           probes=len(probes), deg_probe='0.3/0.5', beta=100.0, L=3, a=0.02,
                           use_resid=False, judge='obs_of (L-035)', smoke=SMOKE),
               T1=results['T1'], T2=results['T2'], T3=results['T3'], gates=gates,
               elapsed_s=round(time.time() - t0, 1))
    suffix = '_smoke' if SMOKE else ''
    with open(os.path.join(RES, 'exp40_public_corpus%s.json' % suffix), 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print('GATES', json.dumps(gates, ensure_ascii=False))
    print('EXP40 DONE %.0fs' % (time.time() - t0), flush=True)

if __name__ == '__main__':
    main()
