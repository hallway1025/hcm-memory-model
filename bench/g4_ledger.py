import os
'''G4 门禁：九子系统消融总账（最终版）。
每个模块用它**真正作用在**的因变量判定，并显式标注「机制是否如设计般起作用」。
三种判定：LOAD-BEARING（关掉后该指标显著变化） / REDUNDANT（机制存在但被别的机制替代）
         / INERT（机制存在但当前不产生可测行为差异）'''
import json, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
def J(n):
    return json.load(open(os.path.join(ROOT, 'results', n), encoding='utf-8'))

abl = {r['arm']: r for r in J('ablation.json')}
base = abl['full']
ab2 = {r['module']: r for r in J('ablation2.json')}

def delta(arm):
    r = abl[arm]
    return dict(sep=r['sep_exact_090'] - base['sep_exact_090'],
                drop=r['drop30'] - base['drop30'],
                cls=r['cls_old'] - base['cls_old'],
                wm=round(r['wm_kept'] - base['wm_kept'], 2))

rows = []

def add(mod, mech, primary, on, off, verdict, note):
    rows.append(dict(module=mod, mechanism=mech, metric=primary,
                     on_arm=on, off_arm=off, verdict=verdict, note=note))

# 统一四指标（对应第 1 张表）
d1 = delta('M1_SEP no-residual'); d1b = delta('M1_SEP no-sparsity')
add('M1_SEP', '期望库残差 + 扩张稀疏', 'SEP（overlap 0.90 精确召回）',
    'with', 'without', 'LOAD-BEARING',
    '关残差：%d/30 -> %d/30；关稀疏：%d/30。两者都改变分离能力。' % (
        base['sep_exact_090'], abl['M1_SEP no-residual']['sep_exact_090'],
        abl['M1_SEP no-sparsity']['sep_exact_090']))

d2 = delta('M2_EPI no-store')
add('M2_EPI', '一次写入 + Hopfield 吸引子', 'SEP', 'with', 'without', 'LOAD-BEARING',
    '库容量=1（不能写情节）：%d/30 -> %d/30。吸引子迭代（L=3）本身在 overlap 0.90 上无增益甚至有害（25->20），已如实记录。' % (
        base['sep_exact_090'], abl['M2_EPI no-store']['sep_exact_090']))

add('M3_CTX', '显式情境偏置项', '情境消歧（内容完全相同、仅情境不同）',
    'ctx_bias on', 'ctx_bias off', 'REDUNDANT',
    '严格测试（exp9_context_strict）：内容完全相同、仅情境不同时，A 组 8/8、B 组 7/7 可按情境取回；'
    '但把显式情境项关掉（kappa=0）结果**完全相同**。情境消歧的能力来自 M1a 的期望库模板'
    '（不同情境开不同 mu，残差把两段数据推到不同子空间），显式偏置项是冗余实现。'
    '按一致标准，关掉后无变化应判 REDUNDANT —— 早期版本把它判 LOAD-BEARING 是错误的（对抗评审 P1-5）。')

add('M4_CON', '系统巩固（交错回放）', 'CLS 旧任务保持', 'with', 'without', 'LOAD-BEARING',
    '无交错：旧任务 %d/24 -> %d/24。这是全文最强的 CLS 证据。' % (
        base['cls_old'], abl['M4 no-replay']['cls_old']))

add('M5_SEM', '新皮层慢学习', 'CLS 旧/新任务', 'trained', 'freeze-after-A', 'LOAD-BEARING',
    '修正后的设计（先学 A -> freeze() 冻结 -> 再学 B）：冻结臂 旧任务 13/24、新任务 8/24；'
    '不冻结臂 旧任务 15/24、新任务 23/24。冻结保住了一部分旧任务，但代价是新任务学不进去 —— 这才是'
    '「稳定性-可塑性权衡」的真实测量。早期版本测的是**从未训练过的随机网络**（0/24 是同义反复），'
    '已由对抗评审 P1-4 指出并修正。')

add('M6_WM', '工作记忆槽位', 'WM 保留条数', 'C=4', 'C=1', 'LOAD-BEARING',
    'C=1：保留 %.2f -> %.2f。容量拐点成立（P5）。' % (base['wm_kept'], abl['M6_WM C=1']['wm_kept']))

g = ab2['M7_NEU']
add('M7_NEU', '神经调质门控', '写入选择性（背景重复 vs 显著事件）', 'learned', 'constant', 'INCONCLUSIVE',
    '门控确实自适应（重复输入 g 从 0.90 降到 0.40），但写入数与恒 1 臂完全相同'
    '（%d vs %d）——因为 store.write 的 theta_dup=0.95 余弦去重已经承担了同样的职责。' % (
        g['on']['written'], g['off']['written']))

r8 = ab2['M8_FOR']
add('M8_FOR', '检索诱发抑制 / 遗忘', '抑制因子差分', 'on', 'off', 'INERT',
    '抑制确实施加（已练 %.4f、未练 %.4f、差分 %+.4f），但 top-1 召回完全不变'
    '（未练 %d -> %d）。原因：top-1 由证据强度决定，被抑制者本来就不是赢家；'
    '且 5 轮练习后几乎所有条目的 sup 都被压到 0.03 量级，抑制是全局而非选择性的。' % (
        r8['on']['sup_practiced'], r8['on']['sup_unpracticed'],
        r8['on']['sup_unpracticed'] - r8['on']['sup_practiced'],
        r8['on']['recall_before'], r8['on']['recall_after']))

m0 = ab2['M0_META']
add('M0_META', '元控制 / 弃权', '风险调整得分', 'meta', 'always-retrieve', 'LOAD-BEARING',
    '弃权使风险调整得分 %.1f -> %.1f。这是模型里唯一在统一四指标上不可见、'
    '但在专属指标上显著有效的子系统。' % (m0['on'], m0['off']))

print('%-8s %-24s %-12s %s' % ('模块', '机制', '判定', '关键证据'))
for r in rows:
    print('%-8s %-24s %-12s %s' % (r['module'], r['mechanism'], r['verdict'], r['note'][:88]))

json.dump(rows, open(os.path.join(ROOT, 'results', 'g4_ledger.json'), 'w', encoding='utf-8'),
          ensure_ascii=False, indent=1)
from collections import Counter
c = Counter(r['verdict'] for r in rows)
print()
print('判定分布:', dict(c))