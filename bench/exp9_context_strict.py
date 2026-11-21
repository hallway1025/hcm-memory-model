'''E9b：M3 严格情境测试——**控制内容完全相同**，只让情境不同。
若模型真的用情境消歧，那么在内容完全相同时，它必须靠情境取回不同条目。
这是排除「残差+内容已足够」这一替代解释的唯一办法。'''
import sys, os, json
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src'))
import numpy as np
from hcm import HCM, Config
from hcm.ops import unit
from hcm.data import random_patterns

D = 128
rows = []
for use_ctx, kappa in ((False, 0.0), (True, 0.0), (True, 1.0), (True, 3.0)):
    cfg = Config(d_x=D, a=0.02, capacity=800, beta=20.0, L=3, use_resid=True, seed=0,
                 use_ctx_bias=use_ctx, kappa_ctx=kappa)
    m = HCM(cfg)
    r = np.random.default_rng(9)
    CONTENT = random_patterns(8, D, seed=33)     # 8 个**同一**内容
    ctxA, ctxB = unit(r.normal(size=D)), unit(r.normal(size=D))
    written = {'A': [], 'B': []}
    for tag, ctx in (('A', ctxA), ('B', ctxB)):
        for _ in range(30):
            m.c = (0.95 * m.c + 0.05 * (1.2 * ctx)).astype(np.float32)
        for c in CONTENT:
            before = m.E.n
            m.observe(unit(c + 1.2 * ctx))
            if m.E.n > before:
                written[tag].append((c, m.E.n - 1))
    # 同一内容在两种情境下各有一条；查询必须靠情境选对
    ok = {'A': 0, 'B': 0}
    tot = {'A': 0, 'B': 0}
    for tag, ctx in (('A', ctxA), ('B', ctxB)):
        for c, idx in written[tag]:
            m.c = (1.2 * ctx).astype(np.float32)
            q = m.query(unit(c + 1.2 * ctx))
            tot[tag] += 1
            ok[tag] += int(q['idx'] == idx)
    rows.append(dict(use_ctx=use_ctx, kappa=kappa, nA=len(written['A']), nB=len(written['B']),
                     A=ok['A'], B=ok['B'], totA=tot['A'], totB=tot['B']))
    print('ctx_bias=%-5s kappa=%-4.1f  写入 A=%d B=%d  情境消歧 A:%d/%d B:%d/%d' % (
        use_ctx, kappa, len(written['A']), len(written['B']), ok['A'], tot['A'], ok['B'], tot['B']))
json.dump(rows, open(os.path.join(ROOT, 'results', 'exp9_context_strict.json'), 'w'), indent=1)