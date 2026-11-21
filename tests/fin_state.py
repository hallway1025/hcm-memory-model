import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
p = os.path.join(ROOT, 'memory', '10-STATE.md')
s = open(p, encoding='utf-8').read()
s = s.replace('- 冻结凭证：audit/SHA256SUMS.txt（94 个文件）',
              '- 冻结凭证：audit/SHA256SUMS.txt（94 个文件），自哈希 BD51A15E4F322F18A0322283927B80B1E1EFDFB80B966513A8729B00DF3D880B')
s = s.replace('- 最终复现：`py -3 bench\\reproduce_all.py` → 18/18 步 PASS，319 秒',
              '- 最终复现：py -3 bench\\reproduce_all.py -> 18/18 步 PASS，232-319 秒（视缓存）')
open(p, 'w', encoding='utf-8').write(s)
print('state finalized')