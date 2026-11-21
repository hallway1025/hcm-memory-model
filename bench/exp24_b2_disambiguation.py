# -*- coding: utf-8 -*-
"""exp24: B2 时间上下文消歧判决实验（peer-01 训练段 OK x N）。

背景：重复短语（如 OK 出现 40+ 次/600 消息）在单时间尺度慢情境（gamma_c=0.95 EMA）
下写入时码几乎相同，theta_dup（0.95）把重复 episode 拒之门外 —— 写侧坍缩。
本实验两问：
  Q1 诊断：重复 episode 的写码/情境相似度结构（写侧坍缩的定量刻画）。
  Q2 判决：强制全写入（绕过 theta_dup，theta_dup=-1）后，
     臂 A_slow: use_ctx_bias + 慢情境 cue（gamma_c=0.95 的 EMA）
     臂 A_fast: 同上但把 E.Ctx 换成快情境（gamma=0.30 的 3 条窗口 EMA）
     臂 B:      无情境偏置（纯内容，基线 = 1/K）
     臂 C:      错误 cue（快情境，特异性：应命中被 cue 的邻居而非目标）
预测：B 约 1/K；A_slow 因写侧情境几乎相同而接近 1/K；A_fast 显著高于 1/K；
C 命中的是被 cue 的 episode（特异性>0.5）。
"""
import os, sys, csv, json, re, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
from hcm.model import HCM, Config

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
PHRASE = os.environ.get('EXP24_PHRASE', 'OK')
NSTREAM = int(os.environ.get('EXP24_NSTREAM', '600'))
MAXEP = int(os.environ.get('EXP24_MAXEP', '40'))
D = int(os.environ.get('EXP24_D', '256'))
GAMMA_FAST = 0.30

_NANO = re.compile(r'\[[^\]]{1,12}\]|@[\w一-龥]{1,16}')
def _clean(s):
    return _NANO.sub('', s).strip()

def main():
    t0 = time.time()
    p = os.path.join(ROOT, 'bench', 'corpus', 'train__peer-01.csv')
    with open(p, encoding='utf-8-sig', newline='') as f:
        rd = csv.reader(f); next(rd)
        rows = [r[2] for r in rd]
    msgs = [_clean(x) for x in rows[:NSTREAM]]
    n_occ = sum(1 for m in msgs if m == PHRASE)
    print('stream', len(msgs), '| phrase occurrences:', n_occ, flush=True)
    vocab = sorted(set(''.join(msgs)) | {'\ufffd'})
    rng = np.random.default_rng(7)
    V = {c: i for i, c in enumerate(vocab)}
    W = rng.standard_normal((D, len(V))).astype(np.float32)
    W /= np.linalg.norm(W, axis=0, keepdims=True) + 1e-9
    def vec(s):
        idx = [V[c] for c in s if c in V] or [V['\ufffd']]
        x = W @ np.bincount(np.asarray(idx, dtype=np.int64), minlength=len(V)).astype(np.float32)
        return (x / (np.linalg.norm(x) + 1e-12)).astype(np.float32)
    X = np.asarray([vec(m) for m in msgs], dtype=np.float32)

    cfg = Config(d_x=D, d_c=D, capacity=6000, beta=100.0, L=0, lam=0.6,
                 use_resid=False, seed=1, use_ctx_bias=True, kappa_ctx=0.6)
    m = HCM(cfg)
    # 快情境（实验侧并行状态，不动 src）
    c_fast = np.zeros(D, dtype=np.float32)
    fastctx_rows = []
    episodes = []
    n_rejected = 0
    for i, x in enumerate(X):
        c_fast = (GAMMA_FAST * c_fast + (1.0 - GAMMA_FAST) * x).astype(np.float32)
        if msgs[i] == PHRASE:
            # 强制写入：走与 observe 相同的编码路径，theta_dup=-1 绕过去重
            x_u, u = m._assemble(x)
            xe = m.enc.resid_write(x_u)
            s_pre = m.enc.pre(xe)
            h = m.enc.encode_from_pre(s_pre)
            before = m.E.n
            uc = (m.c / (np.linalg.norm(m.c) + 1e-12)).astype(np.float32)  # 与 observe 的 ctx=unit(self.c) 一致
            if i >= 3:
                cue_part = (0.125 * X[i - 3] + 0.25 * X[i - 2] + 0.5 * X[i - 1]).astype(np.float32)
                cue_part = (cue_part / (np.linalg.norm(cue_part) + 1e-12)).astype(np.float32)
            else:
                cue_part = uc.copy()
            slot = m.E.write(h, m.t, 1.0, 0.0, 1.0, theta_dup=1.1, payload=x_u, dense=s_pre,
                             ctx=uc, tau=m._timecells())
            if slot >= 0:
                episodes.append({'obs': i, 'slot': int(slot),
                                 'cue_slow': uc.copy(), 'cue_partial': cue_part.copy(),
                                 'code': h.copy()})
                fastctx_rows.append((c_fast / (np.linalg.norm(c_fast) + 1e-12)).astype(np.float32))
                m.wm.present(h); m.t += 1
            else:
                n_rejected += 1
                # write 返回 -1 只可能因容量（theta_dup=-1 不会触发）——记录
        else:
            rec = m.observe(x)
            _ = rec
        if len(episodes) >= MAXEP:
            break
    K = len(episodes)
    print('force-written episodes:', K, '| rejected:', n_rejected, '| E.n:', m.E.n,
          '| time', round(time.time() - t0, 1), 's', flush=True)
    if K < 5:
        print('INSUFFICIENT EPISODES'); return
    fastctx = np.stack(fastctx_rows, axis=1)  # (D, K)
    # ---- Q1 诊断：episode 间码/情境相似度 ----
    codes = np.stack([e['code'] for e in episodes], axis=1)  # (d_h, K)
    Cc = codes.T @ codes
    iu = np.triu_indices(K, 1)
    code_pairs = Cc[iu]
    slow = np.stack([e['cue_slow'] for e in episodes], axis=1)
    slow_pairs = (slow.T @ slow)[iu]
    fast_pairs = (fastctx.T @ fastctx)[iu]
    diag = {'code_pairwise_cos': {'mean': round(float(code_pairs.mean()), 4),
                                   'frac_ge_0.95': round(float((code_pairs >= 0.95).mean()), 4)},
            'cue_slow_pairwise_cos': round(float(slow_pairs.mean()), 4),
            'cue_fast_pairwise_cos': round(float(fast_pairs.mean()), 4),
            'n_occurrences': int(n_occ), 'n_written': int(K)}
    print('DIAG', json.dumps(diag), flush=True)
    # ---- Q2 检索判决 ----
    xq = vec(PHRASE)
    slots = [e['slot'] for e in episodes]
    slot_pos = {s: j for j, s in enumerate(slots)}
    def run_arm(tag, ctx_mode):
        orig = m.E.Ctx.copy()
        if ctx_mode == 'fast':
            m.E.Ctx = fastctx.copy()
        hits = 0; picks = {}
        for j, e in enumerate(episodes):
            m.c = e['cue_slow'].copy()  # 慢 cue 仍注入（A 组两个机制并用；fast 组覆盖 Ctx）
            r = m.query(xq)
            got = slot_pos.get(int(r['idx']), -1)
            if got == j: hits += 1
            picks[str(got)] = picks.get(str(got), 0) + 1
        m.E.Ctx = orig
        return {'hit': hits, 'n': K, 'rate': round(hits / K, 4), 'pick_hist_top': dict(sorted(picks.items(), key=lambda kv: -kv[1])[:5])}
    res = {'phrase': PHRASE, 'K': K, 'diag': diag, 'arms': {}}
    # B: 真关闭偏置
    m.cfg.use_ctx_bias = False
    hits = 0; picks = {}
    for e in episodes:
        m.c = e['cue_slow'].copy()
        r = m.query(xq)
        got = slot_pos.get(int(r['idx']), -1)
        if got == episodes.index(e): hits += 1
        picks[str(got)] = picks.get(str(got), 0) + 1
    m.cfg.use_ctx_bias = True
    res['arms']['B_no_ctx'] = {'hit': hits, 'n': K, 'rate': round(hits / K, 4),
                               'pick_hist_top': dict(sorted(picks.items(), key=lambda kv: -kv[1])[:5])}
    res['arms']['A_slow_cue'] = run_arm('A', 'slow')
    # A_partial: 部分线索（前 3 条消息的截断 EMA，非存储向量本身）—— 海马模式完成检验
    hits = 0
    for e in episodes:
        m.c = e['cue_partial'].copy()
        r = m.query(xq)
        if int(r['idx']) == e['slot']: hits += 1
    res['arms']['A_partial_cue'] = {'hit': hits, 'n': K, 'rate': round(hits / K, 4)}
    # A_tau: 时间细胞臂 —— Ctx := Tau，cue = 写入时刻 tau
    orig = m.E.Ctx.copy()
    m.E.Ctx = m.E.Tau.copy()
    hits = 0; picks = {}
    for j, e in enumerate(episodes):
        m.c = m.E.Tau[:, e['slot']].copy()
        r = m.query(xq)
        got = slot_pos.get(int(r['idx']), -1)
        if got == j: hits += 1
        picks[str(got)] = picks.get(str(got), 0) + 1
    m.E.Ctx = orig
    res['arms']['A_tau'] = {'hit': hits, 'n': K, 'rate': round(hits / K, 4),
                            'pick_hist_top': dict(sorted(picks.items(), key=lambda kv: -kv[1])[:5])}
    # Arm C: 错 cue（慢情境注入邻居 cue；Ctx 本身就是慢情境，无需换列）
    cued = 0; tgt = 0
    for j in range(K):
        j2 = (j + 1) % K
        m.c = episodes[j2]['cue_slow'].copy()
        r = m.query(xq)
        got = slot_pos.get(int(r['idx']), -1)
        if got == j2: cued += 1
        if got == j: tgt += 1
    res['arms']['C_wrong_cue_slow'] = {'cued_rate': round(cued / K, 4), 'target_rate': round(tgt / K, 4), 'n': K}
    print(json.dumps(res['arms'], ensure_ascii=False, indent=1), flush=True)
    json.dump(res, open(os.path.join(ROOT, 'results', 'exp24_b2_disambiguation.json'), 'w'),
              ensure_ascii=False, indent=1)
    print('saved', flush=True)

main()
