import os, json
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
p = os.path.join(ROOT, 'paper', 'PAPER.md')
s = open(p, encoding='utf-8').read()

old = '而且在此数据（随机模式、无共享成分）上，**残差反而更差**（3 vs 8）。这直接说明残差不是万能增益，它只在输入含共享期望成分时有益。'
new = ('关于残差在**随机（无共享成分）数据**上的作用，我们做了 8 种子重测（bench/exp14_resid_harmful.py），'
  + '结果比早期单次测量更细致：' + chr(10) + chr(10)
  + '| N | 精确线索（无残差 / 有残差） | 25% 缺失（无残差 / 有残差） |' + chr(10)
  + '|---|---|---|' + chr(10)
  + '| 4000 | 1.000±0.000 / 1.000±0.000 | 0.997±0.006 / 0.975±0.021 |' + chr(10)
  + '| 8000 | 1.000±0.000 / 1.000±0.000 | **0.938±0.024 / 0.822±0.038** |' + chr(10) + chr(10)
  + '在 N=4000 时两臂的 CI 重叠，**效应不存在**；到 N=8000 才出现分离，且方向确认是残差更差（0.822 vs 0.938，CI 不重叠）。'
  + '所以正确的表述是：**残差的代价随库规模增长而显现**，不是「任何情况下都有害」。早期版本用 N=16000 的单次测量下结论，'
  + '未做多种子验证，由第二轮独立审查指出后补做（审查 P0-1）。')

if old in s:
    s = s.replace(old, new)
    print('section 4.3 residual claim replaced with multi-seed')
else:
    print('WARN: target sentence not found')
    for ln in s.split(chr(10)):
        if '残差反而更差' in ln: print('  found:', ln[:160])

open(p, 'w', encoding='utf-8').write(s)
print('chars:', len(s))