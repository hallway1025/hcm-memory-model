import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
p = os.path.join(ROOT, 'memory', '10-STATE.md')
s = open(p, encoding='utf-8').read()
s = s.replace('## 两个独立 agent 的结论（重要）', '## 三个独立 agent 的结论（重要）')
s = s.replace('**这些已修**（参数统一、该主张已替换为实测 exp11、加了 SHA256 清单）。',
 '**这些已修**（参数统一、该主张已替换为实测 exp11、加了 SHA256 清单）。' + chr(10)
 + '**冻结版洁净复现**（frozen-replicator）：只读三份文档 + 校验和，禁止读源码/跑脚本，自己重实现 L0 三件套。' + chr(10)
 + '结果：清单自哈希匹配、94 文件 0 不一致；有残差臂复现到 26.33±0.50（文档 24.42±1.61）✅；' + chr(10)
 + '**但无残差臂复现到 27.17±0.47（文档 1.08±0.16）❌** —— 主张 (i) 的关键对比复现不出来。' + chr(10)
 + '容量结论与双读出结论复现成功。根因是四个文档缺陷（见论文 §9.6），已全部修正。')
open(p, 'w', encoding='utf-8').write(s)
print('state updated')