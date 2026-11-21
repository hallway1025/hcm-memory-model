#!/usr/bin/env python3
'''exp52_l2_1e6.py -- C2 N=1e6 规模门（SCALE-PROTOCOL L2 收口档，exp37 协议放大 + fastpath 倒排）。

预注册：40-EXP-LEDGER exp52 节（commit c7b3847，2026-11-12，先于任何正式数据）。
协议 = exp37 同款（合成单元模式，theta==0 pinned D-038，judge obs_of L-035，
Config(d_x=2048, d_c=2048, a=0.02, capacity=N+10, beta=100, L=3, use_resid=cell, seed)），
存储路径 = FastEpisodic 倒排（canonical f64 1e6 不可行：写 0.68s/条 x 1e6 ≈ 7.9 天 + Mn 62.5GB）。

modes:
  lockstep  机1 冒烟：真 HCM（canonical EpisodicStore）vs 本脚本 rig+mirror 复刻，
            逐写/逐探针对照 —— 验证复刻代码与 canonical 逐位同行为（含 RIF/sup 漂移）。
  eq        AutoDL 先决门 G-EQ：N=2e5，dense-f32 镜像链 vs postings 链逐探针对照
            （F+T 两 cell seed 0）；门 = hard mismatch=0 且 |da|<1e-4。
  scale     AutoDL：N=1e6 postings only；G-S1 primary（deg25 >= 0.90）+ G-S2 弱化
            （F vs T deg0 |diff|<0.05）+ G-S3'（写入线性）+ RSS/时延曲线。

构造（预注册）：X 分块生成 rng(700+chunk)（chunk=1e5；chunk0 == exp37 X 前 1e5 行）；
探针 = np.linspace(0, N-1, 60).round()（跨库均匀）；degrade 种子 = 41000+i（exp37 用 i，
已预注册偏离）；deg ∈ {0, 0.25}。诚实限制：1e6 单 seed。
'''
import argparse, gc, json, os, sys, time
import numpy as np

try:                                    # Linux（AutoDL）
    import resource
    def peak_rss_mb():
        return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)
except ImportError:                     # Windows（机1 冒烟）
    try:
        import psutil
        def peak_rss_mb():
            return round(psutil.Process().memory_info().rss / 1048576, 1)
    except ImportError:
        def peak_rss_mb():
            return -1

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))
from hcm import HCM, Config                      # noqa: E402
from hcm.ops import unit, softmax_beta           # noqa: E402
from hcm.encoder import Encoder                  # noqa: E402
from hcm.semantic import SemanticStore           # noqa: E402
from hcm.wm import WorkingMemory                 # noqa: E402
from hcm.gate import WriteGate                   # noqa: E402
from hcm.control import Metacontroller           # noqa: E402
from hcm.fastpath import FastEpisodic            # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D_X = 2048
CHUNK = 100000
DEG_SEED_BASE = 41000
THETA_DUP = 0.95
RSS_GUARD_MB = 78000


def pats_full(n, d, seed):
    r = np.random.default_rng(seed)
    return np.asarray([unit(v) for v in r.standard_normal((n, d))], dtype=np.float32)


def pats_chunk(c, d, n_rows):
    r = np.random.default_rng(700 + c)
    return np.asarray([unit(v) for v in r.standard_normal((n_rows, d))], dtype=np.float32)


def degrade(x, frac, seed):
    r = np.random.default_rng(seed)
    y = x.copy()
    nz = np.where(np.abs(y) > 1e-6)[0]
    y[r.choice(nz, size=max(1, int(len(nz) * frac)), replace=False)] = 0.0
    return unit(y)


class Rig:
    """exp37 cell 的 enc/sem/wm/gate/ctl/c/v/t 全套（E 用 ScaleStore 替代；无 Consolidator）。"""

    def __init__(self, cell, N, seed):
        resid = cell.startswith('T')
        c = Config(d_x=D_X, d_c=D_X, a=0.02, capacity=N + 10, beta=100.0, L=3,
                   use_resid=resid, seed=seed)
        self.cfg = c
        self.enc = Encoder(c.d_x, c.d_c, c.d_tau, c.d_h, c.k, seed=c.seed,
                           expand=c.expand, learnable=c.learn_encoder, norm=c.norm, rel=c.rel,
                           homeo=c.homeo, homeo_eta=c.homeo_eta, use_resid=c.use_resid,
                           bank_cap=c.bank_cap, switch_ratio=c.switch_ratio,
                           switch_min_gap=c.switch_min_gap)
        self.sem = SemanticStore(c.d_x, c.d_h, c.d_z, seed=c.seed + 2, kappa=c.kappa, u_meta=c.u_meta)
        self.wm = WorkingMemory(c.d_h, C=c.C)
        self.gate = WriteGate(constant=c.gate_constant)
        self.ctl = Metacontroller(theta_rec=c.theta_rec, margin=c.margin)
        self.c = np.zeros(c.d_c, dtype=np.float32)
        self.v = np.zeros(c.d_h, dtype=np.float32)
        self.t = 0

    def timecells(self):
        c = self.cfg
        j = np.arange(c.d_tau, dtype=np.float32)
        mu = (j + 0.5) / c.d_tau * c.T_ctx
        sigma = c.T_ctx / (2.0 * c.d_tau)
        tt = float(self.t % c.T_ctx)
        tau = np.exp(-((tt - mu) ** 2) / (2.0 * sigma ** 2 + 1e-8)).astype(np.float32)
        return unit(tau)


class ScaleStore:
    """fastpath 倒排 + 可选 dense-f32 镜像 + D 矩阵 + sup/use（append-only，无 prune 路径）。"""

    def __init__(self, d_h, capacity, beta, mirror=False, nnz_hint=328):
        self.d_h, self.capacity = int(d_h), int(capacity)
        self.fast = FastEpisodic(d_h, capacity, k=1, beta=beta)
        self.mirror = np.zeros((d_h, capacity), dtype=np.float32) if mirror else None
        self.d_mat = np.zeros((d_h, capacity), dtype=np.float32)
        self.use = np.zeros(capacity, dtype=np.int64)
        self.n_attempted = 0
        self.n_dup_rejected = 0
        # 增量 CSR（fix O(N^2) rebuild）：预分配 postings 缓冲，append 每写填切片，
        # fast._csr = 前缀视图元组（O(1) 视图，不 copy）。_mtv/_mt_cols 不改。
        self._cnt = 0
        self._buf_cap = int(capacity) * max(1, int(nnz_hint))
        self._pr = np.empty(self._buf_cap, dtype=np.int32)
        self._pv = np.empty(self._buf_cap, dtype=np.float32)
        self._pc = np.empty(self._buf_cap, dtype=np.int32)

    def _csr_append(self, i, h):
        h = np.asarray(h, dtype=np.float32)
        nz = np.flatnonzero(h)
        if nz.size == 0:
            return
        need = self._cnt + int(nz.size)
        if need > self._buf_cap:          # 兜底：nnz 超出 hint（理论不触发，top-k 上限）
            grow = max(need, self._buf_cap * 2)
            pr = np.empty(grow, dtype=np.int32)
            pv = np.empty(grow, dtype=np.float32)
            pc = np.empty(grow, dtype=np.int32)
            if self._cnt:
                pr[:self._cnt] = self._pr[:self._cnt]
                pv[:self._cnt] = self._pv[:self._cnt]
                pc[:self._cnt] = self._pc[:self._cnt]
            self._pr, self._pv, self._pc, self._buf_cap = pr, pv, pc, grow
        self._pr[self._cnt:need] = nz
        self._pv[self._cnt:need] = h[nz]
        self._pc[self._cnt:need] = i
        self._cnt = need
        self.fast._csr = (self._pr[:need], self._pv[:need], self._pc[:need])

    @property
    def n(self):
        return int(self.fast.n)

    def sc(self, h):
        """M.T @ h：dense 镜像 GEMV 或 postings 单 bincount。"""
        if self.mirror is not None and self.n > 0:
            return self.mirror[:, :self.n].T @ h
        if self.n == 0:
            return np.zeros(0, dtype=np.float64)
        return self.fast._mtv(h)

    def mtv(self, v, dense):
        if dense:
            return self.mirror[:, :self.n].T @ v
        return self.fast._mtv(v)

    def append(self, h, t, s_pre, sc_cached=None, dup_check=True):
        self.n_attempted += 1
        if self.n >= self.capacity:
            return -1
        sc = None
        if dup_check:
            sc = sc_cached if sc_cached is not None else (self.sc(h) if self.n > 0 else None)
            if sc is not None and float(sc.max()) >= THETA_DUP:
                self.n_dup_rejected += 1
                return -1
        i = self.n
        if self.mirror is not None:
            self.mirror[:, i] = h.astype(np.float32)
        self.d_mat[:, i] = s_pre.astype(np.float32)
        # 行式索引 _add 跳过（省 4.3GB rowmat + 328 次 python 循环）：
        # _mtv/_mt_cols 只走 _csr；_build_csr 兜底在 _csr 恒非 None 下不触发。
        self._csr_append(i, h)
        self.fast.t[i] = float(t)
        self.fast.sup[i] = 1.0
        self.fast.n = i + 1
        self.fast.n_write_attempts += 1
        return i

    def suppress(self, a, rho_rif=0.15, topk=5):
        """canonical EpisodicStore.suppress 逐行复刻（argsort(-a)[:5]，winner 计 use）。"""
        if self.n == 0:
            return 0
        order = np.argsort(-a)[:topk]
        if len(order) == 0:
            return 0
        winner = order[0]
        others = [i for i in order[1:] if i != winner]
        self.use[winner] += 1
        for i in others:
            self.fast.sup[i] *= (1.0 - rho_rif)
        return len(others)


def observe_rig(rig, store, x, timing=None, audit_stride=None, audit_out=None):
    """canonical HCM.observe 的复刻（E.write 的 dup 门用缓存 sc 一次 bincount 完成）。

    预注册修订（LEDGER 2026-11-12，仅 scale 档 N=1e6）：audit_stride 给定时，
    每 audit_stride 条写一次全量 sc（dup max + eps + g 记入 audit_out），
    其余写不做 sc/门评估、无条件 append —— 依据：实测 sc=10.7ns/posting 使
    1e6 逐写 sc≈20 天不可行；接受性前提=exp37@1e5 同生成器 100% 接受 +
    audit 子样全过 + n_written=N 后验。audit_stride=None 时保持逐写全语义。
    """
    c = rig.cfg
    x = unit(np.asarray(x, dtype=np.float32))
    rig.c = (c.gamma_c * rig.c + (1.0 - c.gamma_c) * x).astype(np.float32)
    tau_b = rig.timecells()
    t0 = time.perf_counter()
    xe = rig.enc.resid_write(x)
    s_pre = rig.enc.pre(xe)
    h = rig.enc.encode_from_pre(s_pre)
    nu = rig.enc.novelty(x, mu=rig.enc.mu)
    audited = (audit_stride is not None) and ((rig.t % audit_stride) == 0)
    do_sc = (store.n > 0) and (audit_stride is None or audited)
    t1 = time.perf_counter()
    if do_sc:
        sc = store.sc(h)
        p = softmax_beta(sc, c.beta)
        eps = float(-np.log(p.max() + 1e-12))
        sc_max = float(sc.max())
    else:
        sc = None
        eps = float(np.log(2.0)) if store.n == 0 else None
        sc_max = None
    t2 = time.perf_counter()
    vh = float(rig.v @ h)
    rpe = 0.0 - vh
    rig.v += (c.eta_v * rpe * h).astype(np.float32)
    activity = float(np.abs(h).sum() / np.sqrt(max(c.k, 1)))
    idx = -1
    if audit_stride is None:
        g = rig.gate(nu, rpe, eps, activity)
        if g >= c.theta_write:
            idx = store.append(h, rig.t, s_pre, sc_cached=sc)
    else:
        # 修订档：无条件 append；audit 子样记录 sc_max/eps/g 作接受性证据
        g = rig.gate(nu, rpe, eps if eps is not None else 0.0, activity) if audited or store.n == 0 else None
        idx = store.append(h, rig.t, s_pre, dup_check=False)
        if audited and audit_out is not None:
            audit_out.append(dict(i=int(rig.t), n=int(store.n), sc_max=sc_max, eps=eps, g=g))
    t3 = time.perf_counter()
    rig.wm.present(h)
    rig.t += 1
    if timing is not None:
        timing[0] += t1 - t0
        timing[1] += t2 - t1
        timing[2] += t3 - t2
    return idx, g, eps, nu, rpe


def query_rig(rig, store, x, path='postings', L=None, do_suppress=True):
    """canonical HCM.query 的复刻（E 调用换 ScaleStore；dense 路径 M@a 用 f32 cast，
    偏差由 lockstep 模式对真 canonical 量化）。"""
    c = rig.cfg
    L = c.L if L is None else int(L)
    x = unit(np.asarray(x, dtype=np.float32))
    if store.n == 0:
        return {"action": "abstain", "reason": "empty store", "m1": 0.0, "m2": 0.0,
                "nu": 0.0, "idx": -1, "pred": None, "n_considered": 0, "reinstated": False}, None
    dense = (path == 'dense')
    best = None
    k_te = 0
    for k_te, xe_t in enumerate(rig.enc.resid_read_all(x)):
        s_pre_t = rig.enc.pre(xe_t)
        s_ev_t = rig.enc.raw(xe_t)
        qq_t = rig.enc.encode_from_pre(s_pre_t)
        ed = store.mtv(s_ev_t, dense)
        es = store.mtv(qq_t, dense)
        ev_t = (c.w_dense * ed / (np.linalg.norm(ed) + 1e-12)
                + (1.0 - c.w_dense) * es / (np.linalg.norm(es) + 1e-12))
        sc_t = float(ev_t.max())
        if best is None or sc_t > best[0]:
            best = (sc_t, ev_t, s_pre_t, qq_t, k_te, s_ev_t)
    _, ev, s_pre, q0, k_te, s_ev = best
    reinstated = bool(k_te > 0)
    a0 = softmax_beta(ev, c.beta)
    if dense:
        Mn = store.mirror[:, :store.n]
        z = unit(np.maximum(s_pre, 0.0))
        a = softmax_beta(Mn.T @ z, c.beta)
        for _ in range(max(0, int(L))):
            z = unit(Mn @ a.astype(np.float32))
            a = softmax_beta(Mn.T @ z, c.beta)
            sa = a.sum()
            if sa > 0:
                a = a / sa
        q_star = z
        a_hop = a
    else:
        q_star, a_hop = store.fast.retrieve(q0, L=L, beta=c.beta, dense=s_pre, f32=False)
    n = store.n
    a = a_hop * store.fast.sup[:n]
    a = a / (a.sum() + 1e-12)
    shat = unit(rig.sem.forward(x))
    s_raw = store.mtv(shat, dense)
    s_z = (s_raw - s_raw.mean()) / (s_raw.std() + 1e-8)
    s_n = 1.0 / (1.0 + np.exp(-s_z))
    a = a / (a.sum() + 1e-12)
    f = c.lam * a + (1.0 - c.lam) * s_n
    order = np.argsort(-f)
    top1 = int(order[0])
    m1 = float(f[top1])
    m2 = float(f[order[1]]) if len(order) > 1 else 0.0
    nu_q = rig.enc.novelty(x, q_star)
    if dense:
        recon = unit(store.d_mat[:, :n] @ a)                     # canonical 口径：f64 上卷 GEMV + unit
    else:
        recon = unit(store.d_mat[:, :n] @ a.astype(np.float32))  # scale 档：f32 GEMV（lockstep 量化偏差）
    raw_evidence = float(unit(s_pre) @ recon)
    action, reason = rig.ctl.decide_retrieval(m1, m2, nu_q, raw_evidence=raw_evidence)
    out = {"action": action, "reason": reason, "m1": m1, "m2": m2, "nu": float(nu_q),
           "raw_evidence": raw_evidence, "idx": -1, "pred": None,
           "n_considered": int(n), "reinstated": reinstated}
    if action in ("recall", "separate"):
        out["idx"] = top1
    if do_suppress and action in ("recall", "separate"):
        store.suppress(a, rho_rif=c.rho_rif)
    return out, a


def run_writes(rig, store, N, cell, seed, mirror_mode, prog_tag, audit_stride=None, audit_out=None):
    """分块生成 X 并写入；返回 (n_written, write_s, chunk_times, timing)。
    audit_stride 非 None（scale 档修订）：逐写免 sc，每 audit_stride 条记 audit 子样。"""
    t_w0 = time.perf_counter()
    timing = [0.0, 0.0, 0.0]
    chunk_times = []
    n_chunks = (N + CHUNK - 1) // CHUNK
    wt_acc, wt_n = 0.0, 0
    for ci in range(n_chunks):
        rows = min(CHUNK, N - ci * CHUNK)
        Xc = pats_chunk(ci, D_X, rows)
        tc0 = time.perf_counter()
        for j in range(rows):
            t1 = time.perf_counter()
            observe_rig(rig, store, Xc[j], timing, audit_stride=audit_stride, audit_out=audit_out)
            wt_acc += time.perf_counter() - t1
            wt_n += 1
        chunk_times.append(round(time.perf_counter() - tc0, 1))
        del Xc
        gc.collect()
        if (ci + 1) * CHUNK % 50000 < CHUNK and (ci + 1) * CHUNK <= N or ci == n_chunks - 1:
            print('%s s%d write %d/%d %.0fs rss=%sMB s/w=%.4f'
                  % (prog_tag, seed, min((ci + 1) * CHUNK, N), N, time.perf_counter() - t_w0,
                     peak_rss_mb(), wt_acc / max(1, wt_n)), flush=True)
    write_s = round(time.perf_counter() - t_w0, 1)
    n_written = store.n
    print('%s s%d written=%d write=%.0fs rss=%sMB (enc %.1fs sc %.1fs append %.1fs)'
          % (prog_tag, seed, n_written, write_s, peak_rss_mb(), timing[0], timing[1], timing[2]),
          flush=True)
    return n_written, write_s, chunk_times, timing, wt_acc / max(1, wt_n)


def probe_rows(N, probe_idx):
    """分块重生成探针行（只留 60 行）。"""
    out = {}
    by_chunk = {}
    for i in probe_idx:
        by_chunk.setdefault(int(i) // CHUNK, []).append(int(i))
    for ci, idxs in by_chunk.items():
        rows = min(CHUNK, N - ci * CHUNK)
        Xc = pats_chunk(ci, D_X, rows)
        for i in idxs:
            out[i] = Xc[i - ci * CHUNK].copy()
        del Xc
        gc.collect()
    return out


def run_probes(rig, store, Xrows, probe_idx, path, cell, seed, do_suppress):
    out = {}
    for frac in (0.0, 0.25):
        tq = time.perf_counter()
        hits = abst = 0
        for i in probe_idx:
            q = Xrows[i] if frac == 0 else degrade(Xrows[i], frac, DEG_SEED_BASE + i)
            o, _a = query_rig(rig, store, q, path=path, do_suppress=do_suppress)
            hits += int(store.fast.t[int(o["idx"])] == i) if int(o["idx"]) >= 0 else 0
            abst += int(o.get("action") == "abstain")
        dt = (time.perf_counter() - tq) / len(probe_idx)
        out['deg%g' % (frac * 100)] = dict(hits=hits, of=len(probe_idx),
                                           rate=round(hits / len(probe_idx), 4),
                                           abstain=abst, s_per_query=round(dt, 3))
        print('%s s%d %s deg%g: %d/%d abst=%d %.3fs/q'
              % (cell, seed, path, frac * 100, hits, len(probe_idx), abst, dt), flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['lockstep', 'eq', 'scale'], required=True)
    ap.add_argument('--cell', choices=['F_b100', 'T_b100'], default='F_b100')
    ap.add_argument('--N', type=int, default=None)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--probes', type=int, default=60)
    ap.add_argument('--audit', action='store_true',
                    help='强制 audit 档（机1 冒烟验证修订路径用；正式 N=1e6 自动启用）')
    args = ap.parse_args()
    N = args.N
    cell, seed = args.cell, args.seed
    tag = '%s N=%d' % (args.mode, N)
    print('exp52 %s cell=%s seed=%d rss0=%sMB' % (tag, cell, seed, peak_rss_mb()), flush=True)

    if args.mode == 'lockstep':
        X = pats_full(N, D_X, 700)
        m = HCM(Config(d_x=D_X, d_c=D_X, a=0.02, capacity=N + 10, beta=100.0, L=3,
                       use_resid=cell.startswith('T'), seed=seed))
        rig = Rig(cell, N, seed)
        store = ScaleStore(rig.cfg.d_h, N + 10, beta=100.0, mirror=True)
        n_mis = 0
        tw = time.perf_counter()
        for i, x in enumerate(X):
            idx, g, eps, nu, rpe = observe_rig(rig, store, x)
            rec = m.observe(x)
            if (rec['wrote'] != (idx >= 0)) or (rec['idx'] != idx and rec['wrote']):
                n_mis += 1
                if n_mis < 5:
                    print('WRITE MISMATCH i=%d canonical idx=%s rig idx=%d' % (i, rec['idx'], idx), flush=True)
        write_s = round(time.perf_counter() - tw, 1)
        print('lockstep writes: canonical n=%d rig n=%d mismatches=%d %.0fs'
              % (int(m.E.n), store.n, n_mis, write_s), flush=True)
        obs_m = {int(m.E.t[j]): j for j in range(int(m.E.n))}
        probe = list(range(0, N, max(1, N // args.probes)))[:args.probes]
        metrics = dict(judge_m=0, judge_r=0, idx_agree=0, action_agree=0, abst_m=0, abst_r=0,
                       max_d_m1=0.0, max_d_m2=0.0, max_d_raw=0.0, max_d_a=0.0, max_d_sup=0.0)
        for frac in (0.0, 0.25):
            for i in probe:
                q = X[i] if frac == 0 else degrade(X[i], frac, DEG_SEED_BASE + i)
                sup0 = store.fast.sup[:store.n].copy()
                om = m.query(q, debug=True)
                a_m = om['dbg']['a']
                orr, _ar = query_rig(rig, store, q, path='dense', do_suppress=False)
                hm = int(obs_m.get(int(om['idx']), -1) == i) if int(om['idx']) >= 0 else 0
                hr = int(store.fast.t[int(orr['idx'])] == i) if int(orr['idx']) >= 0 else 0
                metrics['judge_m'] += hm
                metrics['judge_r'] += hr
                metrics['idx_agree'] += int(om['idx'] == orr['idx'])
                metrics['action_agree'] += int(om['action'] == orr['action'])
                metrics['abst_m'] += int(om['action'] == 'abstain')
                metrics['abst_r'] += int(orr['action'] == 'abstain')
                metrics['max_d_m1'] = max(metrics['max_d_m1'], abs(om['m1'] - orr['m1']))
                metrics['max_d_m2'] = max(metrics['max_d_m2'], abs(om['m2'] - orr['m2']))
                metrics['max_d_raw'] = max(metrics['max_d_raw'], abs(om['raw_evidence'] - orr['raw_evidence']))
                if _ar is not None and len(_ar) == len(a_m):
                    metrics['max_d_a'] = max(metrics['max_d_a'], float(np.max(np.abs(a_m - _ar))))
                if _ar is not None and orr['action'] in ('recall', 'separate'):
                    store.suppress(_ar, rho_rif=rig.cfg.rho_rif)
                sup_m = m.E.sup[:int(m.E.n)]
                sup_r = store.fast.sup[:store.n]
                nn = min(len(sup_m), len(sup_r))
                metrics['max_d_sup'] = max(metrics['max_d_sup'], float(np.max(np.abs(sup_m[:nn] - sup_r[:nn]))))
                # 提交 canonical 侧的 suppress 效果到 rig（下一探针同状态）
                store.fast.sup[:store.n] = sup_m[:store.n]
                del sup0
        gate_pass = bool(n_mis == 0 and metrics['idx_agree'] == 2 * len(probe)
                         and metrics['action_agree'] == 2 * len(probe)
                         and metrics['max_d_a'] < 1e-4 and metrics['max_d_raw'] < 1e-3)
        row = dict(exp='exp52', mode='lockstep', N=N, cell=cell, seed=seed,
                   write_s=write_s, n_write_mismatch=n_mis,
                   n_written_canonical=int(m.E.n), n_written_rig=store.n, probes=len(probe),
                   gate_pass=gate_pass, **metrics)
        with open(os.path.join(ROOT, 'results', 'exp52_lockstep_%s_s%d.json' % (cell, seed)), 'w') as f:
            json.dump(row, f, indent=1)
        print('LOCKSTEP gate_pass=%s metrics=%s' % (gate_pass, json.dumps(metrics)), flush=True)
        return

    if args.mode == 'eq':
        rig = Rig(cell, N, seed)
        store = ScaleStore(rig.cfg.d_h, N + 10, beta=100.0, mirror=True)
        n_written, write_s, chunk_times, timing, spw = run_writes(rig, store, N, cell, seed, True, 'EQ')
        obs_of = {int(store.fast.t[j]): j for j in range(store.n)}
        probe_idx = np.round(np.linspace(0, N - 1, args.probes)).astype(int)
        probe_idx = [int(v) for v in probe_idx]
        Xrows = probe_rows(N, probe_idx)
        print('EQ probes: %d rows regenerated rss=%sMB' % (len(probe_idx), peak_rss_mb()), flush=True)
        cmp = dict(judge_agree=0, action_agree=0, abst_d=0, abst_p=0,
                   max_d_a=0.0, max_d_m1=0.0, max_d_m2=0.0, max_d_raw=0.0, max_d_idx=None)
        hard = 0
        per_probe = []
        for frac in (0.0, 0.25):
            for i in probe_idx:
                q = Xrows[i] if frac == 0 else degrade(Xrows[i], frac, DEG_SEED_BASE + i)
                sup0 = store.fast.sup[:store.n].copy()
                od, ad = query_rig(rig, store, q, path='dense', do_suppress=False)
                op, ap_ = query_rig(rig, store, q, path='postings', do_suppress=False)
                hd = int(obs_of.get(int(od['idx']), -1) == i) if int(od['idx']) >= 0 else 0
                hp = int(obs_of.get(int(op['idx']), -1) == i) if int(op['idx']) >= 0 else 0
                agree = int(od['idx'] == op['idx'] and od['action'] == op['action'])
                if not agree:
                    hard += 1
                cmp['judge_agree'] += int(hd == hp)
                cmp['action_agree'] += agree
                cmp['abst_d'] += int(od['action'] == 'abstain')
                cmp['abst_p'] += int(op['action'] == 'abstain')
                cmp['max_d_m1'] = max(cmp['max_d_m1'], abs(od['m1'] - op['m1']))
                cmp['max_d_m2'] = max(cmp['max_d_m2'], abs(od['m2'] - op['m2']))
                cmp['max_d_raw'] = max(cmp['max_d_raw'], abs(od['raw_evidence'] - op['raw_evidence']))
                if ad is not None and ap_ is not None and len(ad) == len(ap_):
                    cmp['max_d_a'] = max(cmp['max_d_a'], float(np.max(np.abs(ad - ap_))))
                per_probe.append(dict(pid=i, deg=frac, hd=hd, hp=hp, agree=agree))
                store.fast.sup[:store.n] = sup0
                del sup0
                store.suppress(ad, rho_rif=rig.cfg.rho_rif)
        gate = bool(hard == 0 and cmp['max_d_a'] < 1e-4)
        row = dict(exp='exp52', mode='eq', N=N, cell=cell, seed=seed,
                   n_written=n_written, write_s=write_s, chunk_times=chunk_times,
                   s_per_write=round(spw, 4), timing=dict(enc=round(timing[0], 1), sc=round(timing[1], 1),
                                                          app=round(timing[2], 1)),
                   peak_rss_mb=peak_rss_mb(), probes=len(probe_idx), hard_mismatch=hard,
                   gate_G_EQ=gate, **cmp)
        with open(os.path.join(ROOT, 'results', 'exp52_eq_N%d_%s_s%d.json' % (N, cell, seed)), 'w') as f:
            json.dump(row, f, indent=1)
        print('EQ GATE hard=%d max_d_a=%.2e max_d_raw=%.2e -> G_EQ=%s'
              % (hard, cmp['max_d_a'], cmp['max_d_raw'], gate), flush=True)
        return

    if args.mode == 'scale':
        # 预注册修订（LEDGER 2026-11-12）：N=1e6 逐写 sc 实测 10.7ns/posting ≈ 20 天不可行
        # → scale 档改为免 sc 无条件 append + 每 AUDIT_STRIDE 条全量 sc audit 子样。
        # 接受性证据链 = exp37@1e5 同生成器 100% 接受 + audit 全过 + n_written==N 后验。
        audit_stride = 1000 if (N >= 100000 or args.audit) else None
        audit_out = []
        rig = Rig(cell, N, seed)
        store = ScaleStore(rig.cfg.d_h, N + 10, beta=100.0, mirror=False)
        n_written, write_s, chunk_times, timing, spw = run_writes(
            rig, store, N, cell, seed, False, 'SCALE', audit_stride=audit_stride, audit_out=audit_out)
        audit_bad = [a for a in audit_out if a['sc_max'] is not None and a['sc_max'] >= THETA_DUP]
        audit_g_fail = [a for a in audit_out if a['g'] is not None and a['g'] < 0.5]
        print('SCALE audit n=%d bad_dup=%d g_fail=%d' % (len(audit_out), len(audit_bad), len(audit_g_fail)), flush=True)
        if audit_bad or audit_g_fail or n_written != N:
            print('ACCEPTANCE PREMISE VIOLATED written=%d audit_bad=%d g_fail=%d -> invalid'
                  % (n_written, len(audit_bad), len(audit_g_fail)), flush=True)
            sys.exit(4)
        # 写入完成后：增量 CSR 已在位（fast._csr 恒非 None），此处只兜底 + 释放行式残留
        tw0 = time.perf_counter()
        if store.fast._csr is None:
            _ = store.fast._build_csr()
        csr_mb = round((store.fast._csr[0].nbytes + store.fast._csr[1].nbytes
                        + store.fast._csr[2].nbytes) / 1048576, 1)
        store.fast.rowmat = None
        store.fast.rowval = None
        store.fast.rowlen = None
        gc.collect()
        print('SCALE csr=%sMB built %.1fs rss=%sMB' % (csr_mb, time.perf_counter() - tw0, peak_rss_mb()), flush=True)
        if peak_rss_mb() > RSS_GUARD_MB:
            print('RSS GUARD TRIPPED %sMB -> abort' % peak_rss_mb(), flush=True)
            sys.exit(3)
        obs_of = {int(store.fast.t[j]): j for j in range(store.n)}
        probe_idx = [int(v) for v in np.round(np.linspace(0, N - 1, args.probes)).astype(int)]
        Xrows = probe_rows(N, probe_idx)
        out = run_probes(rig, store, Xrows, probe_idx, 'postings', cell, seed, do_suppress=True)
        row = dict(exp='exp52', mode='scale', N=N, d_x=D_X, cell=cell,
                   resid=cell.startswith('T'), seed=seed,
                   audit_stride=audit_stride,
                   audit_rows=audit_out,
                   write_s=write_s, s_per_write=round(spw, 4), chunk_times=chunk_times,
                   timing=dict(enc=round(timing[0], 1), sc=round(timing[1], 1), app=round(timing[2], 1)),
                   n_written=n_written, n_dup_rejected=store.n_dup_rejected,
                   csr_mb=csr_mb, peak_rss_mb=peak_rss_mb(), probes=len(probe_idx), **out)
        with open(os.path.join(ROOT, 'results', 'exp52_N%d_%s_s%d.json' % (N, cell, seed)), 'w') as f:
            json.dump(row, f, indent=1)
        print('DONE', json.dumps({k: row[k] for k in ('N', 'cell', 'seed', 'write_s', 'peak_rss_mb')}), flush=True)


if __name__ == '__main__':
    main()
