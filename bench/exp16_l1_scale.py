#!/usr/bin/env python3
'''exp16_l1_scale.py —— L1 档（d_x=512, d_h=4096, N=20000）跨机器首跑。

L1 是 §14 规模阶梯的第一级外推（8 倍输入维、4 倍情节维、10 倍条目数）。
本脚本回答三件事：(1) L0 的容量结论在 L1 是否保持（精确召回 ~100%）；
(2) 残差臂 vs 无残差臂在 25% 缺失线索下的差异方向是否与 L0 一致；
(3) 实测 L1 的写入/检索耗时与常驻内存，为 L2/L3 外推提供第二个标定点。
4 个种子（L1 单臂写入约 2-4 分钟，全脚本约 40-60 分钟，跑在二号机）。'''
import sys, os, json, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))
import numpy as np
from hcm import HCM, Config
from hcm.ops import unit

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = 512
N = 20000
SEEDS = list(range(4))

def pats(n, d, seed):
    r = np.random.default_rng(seed)
    return np.asarray([unit(v) for v in r.standard_normal((n, d))], dtype=np.float32)

def degrade(x, frac, seed):
    r = np.random.default_rng(seed)
    y = x.copy()
    nz = np.where(np.abs(y) > 1e-6)[0]
    y[r.choice(nz, size=max(1, int(len(nz) * frac)), replace=False)] = 0.0
    return unit(y)

def ci95(v):
    v = np.asarray(v, float)
    return float(v.mean()), float(1.96 * v.std(ddof=1) / np.sqrt(len(v)))

rows = []
print('L1: d_x=%d d_h=%d N=%d seeds=%s' % (D, 8 * D, N, SEEDS), flush=True)
print('%-6s %-8s %-14s %-14s %-10s %-10s' % ('resid', 'exact', 'deg25', 'write_s', 'query_ms', 'rss_mb'), flush=True)
for resid in (False, True):
    ex, dg, wt, qt, rss = [], [], [], [], []
    for s in SEEDS:
        X = pats(N, D, seed=100 + s)
        t0 = time.time()
        m = HCM(Config(d_x=D, d_c=D, a=0.02, capacity=N + 10, beta=20.0, L=3, use_resid=resid, seed=s))
        for x in X:
            m.observe(x)
        wt.append(time.time() - t0)
        probe = list(range(0, N, N // 40))[:40]
        t0 = time.time()
        ex.append(sum(int(m.query(X[i])['idx'] == i) for i in probe) / len(probe))
        qt.append((time.time() - t0) / len(probe) * 1000)
        dg.append(sum(int(m.query(degrade(X[i], 0.25, i))['idx'] == i) for i in probe) / len(probe))
        try:
            import ctypes, ctypes.wintypes as wt2
            class PMC(ctypes.Structure):
                _fields_ = [('cb', ctypes.c_ulong), ('PageFaultCount', ctypes.c_ulong),
                            ('PeakWorkingSetSize', ctypes.c_size_t), ('WorkingSetSize', ctypes.c_size_t),
                            ('QuotaPeakPagedPoolUsage', ctypes.c_size_t), ('QuotaPagedPoolUsage', ctypes.c_size_t),
                            ('QuotaPeakNonPagedPoolUsage', ctypes.c_size_t), ('QuotaNonPagedPoolUsage', ctypes.c_size_t),
                            ('PagefileUsage', ctypes.c_size_t), ('PeakPagefileUsage', ctypes.c_size_t)]
            pmc = PMC(); pmc.cb = ctypes.sizeof(PMC)
            ctypes.windll.psapi.GetProcessMemoryInfo(ctypes.windll.kernel32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb)
            rss.append(pmc.WorkingSetSize / 1e6)
        except Exception:
            rss.append(-1)
    em, ec = ci95(ex); dm, dc = ci95(dg)
    rows.append(dict(resid=resid, exact_mean=round(em, 3), exact_ci=round(ec, 3),
                     deg25_mean=round(dm, 3), deg25_ci=round(dc, 3),
                     write_s=round(float(np.mean(wt)), 1), query_ms=round(float(np.mean(qt)), 1),
                     rss_mb=round(float(np.mean(rss)), 1), seeds=len(SEEDS), N=N, d_x=D, d_h=8 * D))
    print('%-6s %-8s %-14s %-14s %-10.1f %-10.1f' % (resid,
          '%.3f±%.3f' % (em, ec), '%.3f±%.3f' % (dm, dc),
          float(np.mean(wt)), float(np.mean(qt)), float(np.mean(rss))), flush=True)

json.dump(rows, open(os.path.join(ROOT, 'results', 'exp16_l1_scale.json'), 'w', encoding='utf-8'), indent=1)
print('saved results/exp16_l1_scale.json')