'''E9：M3_CTX 情境绑定的正确实验（修正索引错位，L-015 同类）。
设计：同一批内容出现在两种情境下；查询时给出情境线索，要求取回该情境下的那条。
记录每条事件写入时的**真实**库索引，不用 10+k 假设。'''
import sys, os, json
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src'))
import numpy as np
from hcm import HCM, Config
from hcm.ops import unit
from hcm.data import random_patterns

D = 128
rows = []
for use_ctx in (False, True):
    for kappa in (0.0, 1.0, 3.0):
        cfg = Config(d_x=D, a=0.02, capacity=800, beta=20.0, L=3, use_resid=True, seed=0,
                     use_ctx_bias=use_ctx, kappa_ctx=kappa)
        m = HCM(cfg)
        r = np.random.default_rng(7)
        contents = random_patterns(10, D, seed=31)
        ctxA, ctxB = unit(r.normal(size=D)), unit(r.normal(size=D))
        written = []   # (tag, content, 真实库索引)
        for tag, ctx in (('A', ctxA), ('B', ctxB)):
            for _ in range(30):
                m.c = (0.95 * m.c + 0.05 * (1.2 * ctx)).astype(np.float32)
            for c in contents:
                before = m.E.n
                m.observe(unit(c + 1.2 * ctx))
                if m.E.n > before:
                    written.append((tag, c, m.E.n - 1))
        ok = {'A': 0, 'B': 0}
        cnt = {'A': 0, 'B': 0}
        for tag, c, idx in written:
            ctx = ctxA if tag == 'A' else ctxB
            m.c = (1.2 * ctx).astype(np.float32)
            q = m.query(unit(c + 1.2 * ctx))
            cnt[tag] += 1
            ok[tag] += int(q['idx'] == idx)
        rows.append(dict(use_ctx=use_ctx, kappa=kappa, n_written=len(written),
                         A=ok['A'], A_n=cnt['A'], B=ok['B'], B_n=cnt['B']))
        print('ctx_bias=%-5s kappa=%-4.1f  写入%2d  A: %d/%d  B: %d/%d' % (
            use_ctx, kappa, len(written), ok['A'], cnt['A'], ok['B'], cnt['B']))
json.dump(rows, open(os.path.join(ROOT, 'results', 'exp9_context.json'), 'w'), indent=1)